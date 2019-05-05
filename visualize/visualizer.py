import torch
import numpy as np
import cv2
import os


class Visualizer:
    def __init__(self):
        pass

    def draw(self, target_seq: torch.Tensor, pred_seq: torch.Tensor, h, w, target_seq_dir):
        assert len(target_seq) == len(pred_seq), 'the seq of two must be same'
        assert target_seq.shape[1:] == (68, 2) and pred_seq.shape[1:] == (68, 2)
        if not os.path.exists(target_seq_dir):
            os.makedirs(target_seq_dir)

        for i in range(pred_seq.numpy()[0]):
            np.savetxt(target_seq_dir + '{:0>4d}.jpg.txt'.format(i), pred_seq.numpy(), delimiter=',')
            part_labels = np.zeros((h, w, 3), np.uint8)
            part_labels = self.polylines_keypoints(target_seq.numpy(), part_labels, (0, 255, 0))
            part_labels = self.polylines_keypoints(pred_seq.numpy(), part_labels, (0, 0, 255))
            cv2.imshow('w', part_labels)
            cv2.waitKey(0)


    def polylines_keypoints(self, keypoints, part_labels, color):
        # mapping from keypoints to face part
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                     [range(17, 22)],  # right eyebrow
                     [range(22, 27)],  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],  # nose
                     [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                     [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                     [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                     [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                     ]
        # label_list = [1, 2, 2, 3, 4, 4, 5, 6]  # labeling for different facial parts
        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
        upper_pts = pts[1:-1, :].copy()
        upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

        # label map for facial part
        for p, edge_list in enumerate(part_list):
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            print(pts)
            cv2.polylines(part_labels, pts=[pts], isClosed=True, color=color)

        return part_labels

if __name__ == '__main__':
    pass