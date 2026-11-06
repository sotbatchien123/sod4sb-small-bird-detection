"""Independent data-method views for batch-8 MVA fine-tuning: photometric, tile slicing, bbox paste.

Every arm trains on native-resolution 1280x736 views (the Drone pretrain v2 geometry); a box cut by
the view edge is kept if at least half its area remains. Augmentation and Copy-Paste take one view
per source image per epoch, centred on one random bird (uniform if none). Slicing follows the DL
team (MVA2023): every grid tile (overlap from the recipe) that keeps a bird, plus one clean
background tile per source per epoch. Combined uses the Slicing sample plan and applies both the
Copy-Paste and the photometric step of the other two arms to every tile (same parameters).

All donors come from audited TRAIN annotations. Bbox Copy-Paste is not mask Copy-Paste; patches
are pasted inside the view.
"""
import hashlib
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from yolo_scale_views import intersect, tile_windows, xyxy

VIEW = (1280, 736)  # width, height; must equal the Drone pretrain v2 crop
KEEP_FRACTION = .5


def intersects(boxes, window):
    return intersect(boxes, window)[1] > 0


def kept_mask(boxes, window):
    """Boxes keeping at least KEEP_FRACTION of their area inside window."""
    _, visible = intersect(boxes, window)
    return (visible > 0) & (visible >= KEEP_FRACTION*(boxes[:, 2:]-boxes[:, :2]).prod(axis=1))


def crop_start(rng, lo_edge, hi_edge, size, crop):
    """Window start so that [lo_edge, hi_edge] lies inside the crop; any overlap if the box is larger.
    Same rule as train_yolo_drone.crop_start (Drone pretrain v2)."""
    lo, hi = max(0., hi_edge-crop), min(lo_edge, size-crop)
    if lo > hi: lo, hi = hi, lo
    hi = int(math.floor(min(max(hi, 0), size-crop))); lo = min(int(math.ceil(min(max(lo, 0), size-crop))), hi)
    return int(rng.randint(lo, hi+1))


class PilotViews:
    def __init__(self, data, annotation, recipe):
        self.data, self.recipe = Path(data), recipe
        self.method = recipe['method']
        if self.method not in ('augmentation', 'slicing', 'copy_paste', 'combined'):
            raise ValueError('Unsupported pilot method')
        combined = self.method == 'combined'
        self.uses_tiles = self.method == 'slicing' or combined
        self.uses_paste = self.method == 'copy_paste' or combined
        self.uses_photometric = self.method == 'augmentation' or combined
        if tuple(recipe['view_size']) != VIEW:
            raise ValueError('View size must match the Drone pretrain crop')
        self.by_image = {}
        for a in annotation['annotations']:
            if not a.get('iscrowd', 0):
                self.by_image.setdefault(a['image_id'], []).append(a)
        self.images = {im['id']: im for im in annotation['images']}
        self.donors = []
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        if self.uses_paste:
            for image_id in sorted(self.images):
                im = self.images[image_id]
                anns = sorted(self.by_image.get(image_id, []), key=lambda a:a['id'])
                boxes = xyxy(anns)
                for index, (ann, box) in enumerate(zip(anns, boxes)):
                    bw, bh = box[2:] - box[:2]
                    if min(bw,bh) < 2 or max(bw,bh) > recipe['max_donor_side']:
                        continue
                    pad=recipe['patch_margin']
                    window=(max(0,math.floor(box[0])-pad),max(0,math.floor(box[1])-pad),
                            min(im['width'],math.ceil(box[2])+pad),min(im['height'],math.ceil(box[3])+pad))
                    if not (window[0] <= box[0] < box[2] <= window[2] and window[1] <= box[1] < box[3] <= window[3]):
                        continue
                    hits=intersects(boxes,window)
                    if hits.sum() != 1 or not hits[index]:
                        continue  # Do not paste an unlabelled neighbouring bird.
                    self.donors.append(dict(image_id=image_id,annotation_id=ann['id'],window=list(window),box=box.tolist()))
            if not self.donors:
                raise RuntimeError('No valid TRAIN-only donor patches')
        self.donor_hash=hashlib.sha256(json.dumps(self.donors,sort_keys=True).encode()).hexdigest()

    def tiles(self, im):
        """Slicing plan for one source: (tiles keeping a bird, clean tiles without any bird pixel).
        Tiles that only contain birds cut below KEEP_FRACTION are never used."""
        boxes = xyxy(self.by_image.get(im['id'], []))
        positive, clean = [], []
        for tile in tile_windows(im['width'], im['height'], VIEW, self.recipe['overlap']):
            window = np.array(tile, dtype=np.float32)
            if kept_mask(boxes, window).any(): positive.append(list(tile))
            elif not intersects(boxes, window).any(): clean.append(list(tile))
        return positive, clean

    def donor_image(self,image_id):
        # Views are built in worker threads: guard the cache, decode outside the lock.
        with self.lock:
            if image_id in self.cache:
                self.cache.move_to_end(image_id)
                return self.cache[image_id]
        im=self.images[image_id]
        raw=cv2.imread(str(self.data/'images'/im['file_name']))
        if raw is None or raw.shape[:2] != (im['height'],im['width']):
            raise RuntimeError('Invalid donor image '+im['file_name'])
        with self.lock:
            self.cache[image_id]=raw
            while len(self.cache)>2: self.cache.popitem(last=False)
        return raw

    def __call__(self,raw,boxes,seed,image_id,tile=None):
        """tile: slicing window [x0,y0,x1,y1] chosen by the runner; None for the other methods."""
        rng=np.random.RandomState(seed)
        h,w=raw.shape[:2]; vw,vh=VIEW
        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
        if self.uses_tiles:
            if tile is None: raise ValueError('Slicing needs a tile window')
            x0,y0=int(tile[0]),int(tile[1])
        elif tile is not None: raise ValueError('Only slicing/combined take a tile window')
        elif len(boxes):
            b=boxes[rng.randint(len(boxes))]
            x0=crop_start(rng,b[0],b[2],w,vw); y0=crop_start(rng,b[1],b[3],h,vh)
        else:
            x0=int(rng.randint(0,w-vw+1)); y0=int(rng.randint(0,h-vh+1))
        window=np.array([x0,y0,x0+vw,y0+vh],dtype=np.float32)
        if window[2]>w or window[3]>h: raise ValueError('View exceeds image')
        keep=kept_mask(boxes,window)
        local=intersect(boxes,window)[0]-window[[0,1,0,1]]
        # Every bird pixel inside the view, labelled or dropped, blocks paste placement.
        blocked=local[intersects(boxes,window)]
        kept=local[keep]
        work=raw[y0:y0+vh,x0:x0+vw]; pasted=[]
        if self.uses_paste and rng.rand()<self.recipe['paste_probability']:
            work=work.copy(); occupied=[]
            requested=rng.randint(1,self.recipe['max_patches']+1)
            for _ in range(requested):
                for attempt in range(self.recipe['placement_attempts']):
                    donor=self.donors[rng.randint(len(self.donors))]
                    if donor['image_id']==image_id: continue
                    dx0,dy0,dx1,dy1=donor['window']; pw,ph=dx1-dx0,dy1-dy0
                    x,y=rng.randint(vw-pw+1),rng.randint(vh-ph+1)
                    target=(x,y,x+pw,y+ph)
                    if intersects(blocked,target).any(): continue
                    if occupied and intersects(np.asarray(occupied,dtype=np.float32),target).any(): continue
                    donor_raw=self.donor_image(donor['image_id'])
                    work[y:y+ph,x:x+pw]=donor_raw[dy0:dy1,dx0:dx1]
                    newbox=np.asarray(donor['box'],dtype=np.float32)+np.array([x-dx0,y-dy0,x-dx0,y-dy0],dtype=np.float32)
                    kept=np.concatenate((kept,newbox[None]))
                    occupied.append(target)
                    pasted.append(dict(donor_image_id=donor['image_id'],donor_annotation_id=donor['annotation_id'],
                                       target_window=[x+x0,y+y0,x+x0+pw,y+y0+ph],new_box=(newbox+window[[0,1,0,1]]).tolist()))
                    break
        canvas=work
        # Same basic photometric/geometric augmentation in every arm.
        flipped=bool(rng.rand()<.5)
        if flipped:
            canvas=canvas[:,::-1]
            kept[:,[0,2]]=vw-kept[:,[2,0]]
        canvas=np.clip(canvas.astype(np.float32)*rng.uniform(.8,1.2),0,255).astype(np.uint8)
        photo=False
        if self.uses_photometric and rng.rand()<self.recipe['photometric_probability']:
            hsv=cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0]=(hsv[:,:,0]+rng.uniform(-self.recipe['hue_shift'],self.recipe['hue_shift']))%180
            hsv[:,:,1]=np.clip(hsv[:,:,1]*rng.uniform(*self.recipe['saturation_range']),0,255)
            canvas=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            gamma=rng.uniform(*self.recipe['gamma_range'])
            lut=np.round(255*(np.arange(256,dtype=np.float32)/255)**gamma).astype(np.uint8)
            canvas=cv2.LUT(canvas,lut);photo=True
        norm=kept.copy()
        norm[:,:2]=(kept[:,:2]+kept[:,2:])/2
        norm[:,2:]=kept[:,2:]-kept[:,:2]
        norm/=np.array([vw,vh,vw,vh],dtype=np.float32)
        if not np.isfinite(norm).all() or not ((norm>=0)&(norm<=1)).all() or not (norm[:,2:]>0).all():
            raise RuntimeError('Invalid transformed boxes')
        # dropped: birds partly inside the view but cut below KEEP_FRACTION (birds fully outside are not counted).
        info=dict(window=window.astype(int).tolist(),retained_boxes=len(norm),source_boxes=len(boxes),
                  dropped_boxes=int(len(blocked)-keep.sum()),keep=keep.tolist(),
                  pasted_count=len(pasted),pasted=pasted,photometric_applied=photo,flipped=flipped)
        return np.ascontiguousarray(canvas),norm,info
