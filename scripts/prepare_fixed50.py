"""Version annotations without modifying the original competition split."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def prepare():
    dest = ROOT/'data/mva_audited_v1/annotations'
    dest.mkdir(parents=True, exist_ok=True)
    report = {'version': 1, 'split_changed': False, 'splits': {}}
    for split in ('train', 'val'):
        src = ROOT/'mva2023_sod4bird_train/annotations'/f'split_{split}_coco.json'
        raw = json.loads(src.read_text(encoding='utf-8'))
        seen, kept, removed = {}, [], []
        for a in raw['annotations']:
            key = (a['image_id'], a['category_id'], tuple(a['bbox']), a.get('iscrowd', 0))
            if key in seen:
                removed.append({'removed_id': a['id'], 'kept_id': seen[key], 'image_id': a['image_id'], 'bbox': a['bbox']})
            else:
                seen[key] = a['id']; kept.append(a)
        raw['annotations'] = kept
        target = dest/src.name
        target.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        report['splits'][split] = {'original_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'clean_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'images': len(raw['images']), 'annotations': len(kept), 'removed': removed}
    assert report['splits']['train']['removed'] == []
    assert [a['removed_id'] for a in report['splits']['val']['removed']] == [5704]
    (dest.parent/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': prepare()
