import argparse, json
import os

import parallel_to_frames as parallel


def process_videos(source_directory, target_directory, num_workers, failed_save_file):
    """
    Extract video frames for a class.
    :param num_workers:           Number of worker processes.
    :param failed_save_file:      Path to a log of failed extractions.
    :return:                      None.
    """
    pool = parallel.Pool(source_directory, target_directory, num_workers, failed_save_file)
    pool.start_workers()
    pool.feed_videos()
    pool.stop_workers()


def main(args):
    process_videos(args.root, args.root, args.num_workers, args.failed_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract individual frames from videos for faster loading.")

    parser.add_argument('--root', type=str, required=True, help='where the video put and frames to put')
    parser.add_argument("--num_workers", type=int, default=1, help="number of worker threads")
    parser.add_argument("--failed_log", default="video_to_frames_log.txt", help="where to save list of videos for "
                                                                                "which the frame extraction failed")

    parsed = parser.parse_args()
    main(parsed)
