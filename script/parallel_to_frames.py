import os
import subprocess
from multiprocessing import Process, Queue
from face_landmarks_detection import FaceAligner
import video


class Pool:
    """
    A pool of video downloaders.
    """

    def __init__(self, root_directory, mode, num_workers, failed_save_file):
        self.source_directory = os.path.join(root_directory, mode + '_img')
        self.target_directory = self.source_directory
        self.keypoints_path = os.path.join(root_directory, mode + '_keypoints')
        self.mode = mode
        self.num_workers = num_workers
        self.failed_save_file = failed_save_file

        self.videos_queue = Queue(100)
        self.failed_queue = Queue(100)

        self.workers = []
        self.failed_save_worker = None

    def feed_videos(self):
        """
        Feed videos to a queue for workers.
        :return:      None.
        """
        for root, directorys, filenames in os.walk(self.source_directory):
            for filename in filenames:
                if filename.endswith(('mp4', 'mov')):
                    video_path = os.path.join(root, filename)
                    rela_path = os.path.relpath(root, self.source_directory)
                    save_dir_path = os.path.join(self.target_directory, rela_path, ".".join(filename.split(".")[:-1]))
                    self.videos_queue.put((video_path, save_dir_path))

    def start_workers(self):
        """
        Start all workers.
        :return:    None.
        """

        # start failed videos saver
        if self.failed_save_file is not None:
            self.failed_save_worker = Process(target=write_failed_worker,
                                              args=(self.failed_queue, self.failed_save_file))
            self.failed_save_worker.start()

        # start download workers
        for _ in range(self.num_workers):
            worker = Process(target=video_worker, args=(self.videos_queue, self.failed_queue, self.keypoints_path))
            worker.start()
            self.workers.append(worker)

    def stop_workers(self):
        """
        Stop all workers.
        :return:    None.
        """

        # send end signal to all download workers
        for _ in range(len(self.workers)):
            self.videos_queue.put(None)

        # wait for the processes to finish
        for worker in self.workers:
            worker.join()

        # end failed videos saver
        if self.failed_save_worker is not None:
            self.failed_queue.put(None)
            self.failed_save_worker.join()


def video_worker(videos_queue, failed_queue, keypoints_path):
    """
    Process video files.
    :param videos_queue:      Queue of video paths.
    :param failed_queue:      Queue for failed videos.
    :return:                  None.
    """
    aligner = FaceAligner()

    while True:
        request = videos_queue.get()

        if request is None:
            break

        video_path, save_path = request

        if os.path.isdir(save_path):
            continue

        os.makedirs(save_path)
        if video_path.endswith('mov'):
            converted_path = ".".join(video_path.split(".")[:-1] + ['mp4'])
            if not video.video_to_mp4(video_path, converted_path):
                failed_queue.put(video_path)
                continue
            else:
                os.remove(video_path)
                video_path = converted_path

        if (not video.video_has_sound(video_path)) or \
                (not video.video_to_sound(video_path, ".".join(video_path.split(".")[:-1] + ['wav']))):
            failed_queue.put(video_path)

        if not video.video_to_jpgs(video_path, save_path, do_resize=False):
            failed_queue.put(save_path)

        if not aligner.translate_to_landmarks(save_path, keypoints_path):
            failed_queue.put(save_path)



def write_failed_worker(failed_queue, failed_save_file):
    """
    Write failed video ids into a file.
    :param failed_queue:        Queue of failed video ids.
    :param failed_save_file:    Where to save the videos.
    :return:                    None.
    """

    with open(failed_save_file, "a") as file:
        while True:
            save_path = failed_queue.get()

            if save_path is None:
                break

            file.write("{}\n".format(save_path))
