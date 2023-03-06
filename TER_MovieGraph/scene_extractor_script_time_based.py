# %%
# !pip install ffmpeg-python
import ffmpeg
import pandas as pd
import numpy  as np
import argparse
import os
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description='A script to accept parameters')
parser.add_argument('second_skips', help='The number of seconds skipped between frames', default = 0.5)
args = parser.parse_args()
second_skips = float(args.second_skips)
# print('Second Skips:', second_skips)

# %%
vid_events_cols = ['end_frame', 'end_time', 'cut']
vid_events = pd.read_csv('Shot_and_Scene/video_boundaries/tt0073486.videvents', 
                        delim_whitespace=True, header=0, names=vid_events_cols)
vid_events['shot_id'] = range(1, len(vid_events)+1)
vid_events['start_time'] = vid_events['end_time'].shift(1)
vid_events['start_frame'] = vid_events['end_frame'].shift(1)
vid_events.fillna(0, inplace=True)

# %%
scene_cols = ['start_shot_id', 'end_shot_id', 'valid']
scenes = pd.read_csv('Shot_and_Scene/scene_boundaries/tt0073486.scenes.gt', delim_whitespace=True, header=None, names=scene_cols)

# %%
scene_times = scenes.join(vid_events[['shot_id', 'start_time', 'start_frame']].set_index('shot_id'), on='start_shot_id')
scene_times = scene_times.join(vid_events[['shot_id', 'end_time', 'end_frame']].set_index('shot_id'), on='end_shot_id')
scene_times['end_frame'].fillna(vid_events['end_frame'].max(), inplace=True)
scene_times['end_time'].fillna(vid_events['end_time'].max(), inplace=True)
scene_times[['start_frame', 'end_frame']] = scene_times[['start_frame', 'end_frame']].astype(int)
scene_times['scene_id'] = range(1, len(scene_times)+1)
scene_times.index = scene_times['scene_id']

#movie starts with a 4 second delay
delay = 4
scene_times['start_time'] = scene_times['start_time'] + delay
scene_times['end_time'] = scene_times['end_time'] + delay
scene_times.to_csv('scenes_defined.csv', index=False)

# %%
def saveTimeRangeFrames(probe, file_path, start_time, end_time, out = "", second_skips = 1):
    start_time = int(start_time)
    end_time = int(end_time)
    width = probe['streams'][0]['width']

    if not os.path.exists(out):
        os.makedirs(out)
    else:
        print(f"{file_path} already exists")
        return

    for i in np.arange(start_time, end_time,second_skips):
        out_path = out + 'Image' + str(i).zfill(5) + '.jpg'
        print(out_path)
        if not os.path.exists(out_path):
            ffmpeg.input(file_path, ss=i).filter('scale', width, -1).output(out_path, vframes=1).run()

def saveTimeRangeVid(file_path, start_time, end_time, out = ""):
    start_time = int(start_time)
    end_time = int(end_time)
    if not os.path.exists("scene_library/videos/"):
        os.makedirs("scene_library/videos/")
        
    if not os.path.exists(out):  
        print(f"Making video {out}")
        print(type(start_time), type(end_time))
        trim_video(file_path, out, start_time, end_time)
#         ffmpeg.input(file_path).trim(start=start_time, end=end_time).output(out).run()
        
def trim_video(input_file, output_file, start_time, end_time):
    # create a stream object for the input video
    input_stream = ffmpeg.input(input_file)

    # trim the video from the start_time to the end_time
    trimmed_stream = input_stream.trim(start=start_time, end=end_time)

    # apply the setpts filter to start the trimmed video at time 0
    filtered_stream = trimmed_stream.filter('setpts', 'PTS-STARTPTS')

    # create an output object and save the video to the output file
    output = ffmpeg.output(filtered_stream, output_file, vcodec='libx264')
    ffmpeg.run(output)
    
# %%
movie_file = "tt0073486_One.Flew.Over.the.Cuckoos.Nest.1975.FRENCH.BRRip.x264-Wawacity.cc.mp4"
probe = ffmpeg.probe(movie_file)

for i in tqdm(scene_times['scene_id'].tolist()):
    scene = scene_times[scene_times['scene_id'] == i]
    if scene['valid'].values[0] == 1:
        start_time = scene['start_time'].values[0]
        end_time = scene['end_time'].values[0]
        saveTimeRangeFrames(probe, movie_file, start_time, end_time, out = f"scene_library/frames/scene_{str(i).zfill(3)}_valid.mp4/", second_skips=second_skips)
        saveTimeRangeVid(movie_file, start_time, end_time, out = f"scene_library/videos/scene_{str(i).zfill(3)}_valid.mp4")
    else:
        start_time = scene['start_time'].values[0]
        end_time = scene['end_time'].values[0]
        saveTimeRangeFrames(probe, movie_file, start_time, end_time, out = f"scene_library/frames/scene_{str(i).zfill(3)}_invalid.mp4/", second_skips=second_skips)
        saveTimeRangeVid(movie_file, start_time, end_time, out = f"scene_library/videos/scene_{str(i).zfill(3)}_invalid.mp4")

