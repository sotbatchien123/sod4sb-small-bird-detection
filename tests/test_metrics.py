"""Synthetic software checks of the evaluation and box-fusion code; these are NOT benchmark experiments.

    python -m unittest discover tests
"""
import json, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from evaluate import evaluate          # noqa: E402
import ensemble_wbf, merge_tune        # noqa: E402

# Two 3840x2160 frames; frame 1 has two birds, frame 2 has one. Boxes are COCO [x, y, w, h].
GT = {1: [[100, 100, 20, 20], [500, 300, 12, 12]], 2: [[1000, 800, 30, 30]]}


def write_ann(path):
    ann = dict(images=[dict(id=i, width=3840, height=2160, file_name=f'{i}.jpg') for i in GT],
               categories=[dict(id=0, name='bird')],
               annotations=[dict(id=k, image_id=i, category_id=0, bbox=b, area=b[2]*b[3], iscrowd=0)
                            for k, (i, b) in enumerate(((i, b) for i in GT for b in GT[i]), 1)])
    Path(path).write_text(json.dumps(ann), encoding='utf-8')


def pred(i, b, s=.9):
    return dict(image_id=i, category_id=0, bbox=list(map(float, b)), score=s)


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ann = Path(cls.tmp.name)/'ann.json'
        write_ann(cls.ann)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_perfect_predictions(self):
        m = evaluate([pred(i, b) for i in GT for b in GT[i]], self.ann)
        for k in ('AP50', 'AP75', 'AR100'):
            self.assertAlmostEqual(m[k], 1.0, places=6, msg=k)

    def test_iou_between_050_and_075(self):
        # A 20x20 box shifted by 4 px: IoU = 16*20 / (2*400 - 320) = 0.667 -> counts at IoU 0.5, not at 0.75.
        preds = [pred(1, [104, 100, 20, 20]), pred(1, GT[1][1]), pred(2, GT[2][0])]
        m = evaluate(preds, self.ann, image_ids=[1])
        self.assertAlmostEqual(m['AP50'], 1.0, places=6)
        self.assertLess(m['AP75'], 0.6)

    def test_missed_bird_lowers_recall(self):
        m = evaluate([pred(1, GT[1][0]), pred(2, GT[2][0])], self.ann)
        self.assertAlmostEqual(m['AR100'], 2/3, places=6)

    def test_empty_predictions(self):
        self.assertEqual(evaluate([], self.ann), dict(AP50=0., AP75=0., AR100=0.))

    def test_size_bucket(self):
        # only the 12 px bird lies in the 8-16 px bucket; predicting just that bird gives AP50 = 1 there
        m = evaluate([pred(1, GT[1][1])], self.ann, area_rng=(8, 16))
        self.assertAlmostEqual(m['AP50'], 1.0, places=6)


class FusionTests(unittest.TestCase):
    box = np.array([[100., 100., 120., 120., .9]])

    def test_ensemble_wbf_agreement_keeps_score(self):
        out = ensemble_wbf.wbf([self.box, self.box, self.box], [1., 1., 1.], .55)
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0], self.box[0])

    def test_ensemble_wbf_single_source_keeps_one_third(self):
        empty = np.zeros((0, 5))
        out = ensemble_wbf.wbf([self.box, empty, empty], [1., 1., 1.], .55)
        self.assertAlmostEqual(out[0, 4], .9/3)

    def test_ensemble_wbf_weighted_coordinates(self):
        a = np.array([[100., 100., 120., 120., .8]]); b = np.array([[102., 100., 122., 120., .4]])
        out = ensemble_wbf.wbf([a, b], [1., 1.], .55)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0, 0], (100*.8 + 102*.4)/1.2)      # score-weighted mean
        self.assertAlmostEqual(out[0, 4], (.8 + .4)/2)

    def test_ensemble_wbf_separate_birds_stay_separate(self):
        far = np.array([[500., 300., 512., 312., .7]])
        out = ensemble_wbf.wbf([self.box, far], [1., 1.], .55)
        self.assertEqual(len(out), 2)

    def test_merge_tune_wbf_divides_by_voters(self):
        out = merge_tune.wbf(np.vstack([self.box, self.box]), voters=2)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0, 4], .9)


if __name__ == '__main__':
    unittest.main()
