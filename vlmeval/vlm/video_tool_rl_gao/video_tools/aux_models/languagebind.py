# mainly copied from vdr https://github.com/yhy-2000/VideoDeepResearch#
import os
from .languagebind_models import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer, LanguageBindVideoTokenizer
import torch
import numpy as np
import cv2
import pickle
import time
import json
try:
    from moviepy import VideoFileClip, concatenate_videoclips # for moviepy==1.0.3, for others "from moviepy import VideoFileClip, concatenate_videoclips"
except:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
from decord import VideoReader, cpu
from tqdm import tqdm

import math
import argparse
# from video_utils import *
import subprocess
import datetime
import multiprocessing
import re

from FlagEmbedding import BGEM3FlagModel
from concurrent.futures import ThreadPoolExecutor, as_completed

def _seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def _cut_video_clips(video_path, clips_dir, video_name, duration):    # Check if clips already exist
    existing_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    expected_clips = int(duration // 10) + (1 if duration % 10 > 0 else 0)
    if len(existing_clips) >= expected_clips - 2:
        print(f"Skipping cutting: {len(existing_clips)} clips already exist in {clips_dir}")
        return

    """Cut video into 10-second clips using multithreading"""
    def cut_single_clip(start_time, end_time, clip_index):
        # Format time strings
        start_str = _seconds_to_time_str(start_time)
        end_str = _seconds_to_time_str(end_time)
        
        # Output filename
        output_file = os.path.join(
            clips_dir, 
            f"clip_{clip_index}_{start_str.replace(':', '-')}_to_{end_str.replace(':', '-')}.mp4"
        )
        
        # FFmpeg command to cut clip
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            output_file,
            '-y'  # Overwrite if exists
        ]
        
        subprocess.run(cmd, capture_output=True)
        print(f"Created clip: {os.path.basename(output_file)}")
    
    # Calculate clip intervals
    clip_duration = 10  # seconds
    num_clips = int(duration // clip_duration) + (1 if duration % clip_duration > 0 else 0)
    
    # Use ThreadPoolExecutor for parallel clip cutting
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, duration)
            
            future = executor.submit(cut_single_clip, start_time, end_time, i)
            futures.append(future)
        
        # Wait for all clips to be processed
        for future in as_completed(futures):
            future.result()


def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
    return duration

def preprocess_video_with_progress(video_path, dataset_folder):
    """带进度显示的视频预处理"""
    video_name = video_path.split('/')[-1][:-4]
    
    # 创建输出目录
    clips_dir = os.path.join(dataset_folder, 'clips', '10', video_name)
    frames_dir = os.path.join(dataset_folder, 'dense_frames', video_name)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    
    try:
        # 获取视频时长
        duration = _get_video_duration(video_path)
        
        # 并行处理视频切分和帧提取
        with ThreadPoolExecutor(max_workers=24) as executor:
            clip_future = executor.submit(_cut_video_clips, video_path, clips_dir, video_name, duration)
            
            clip_future.result()
        
    except Exception as e:
        raise RuntimeError(f"Error during video preprocessing: {e}")

def is_valid_video(path):
    try:
        cap = cv2.VideoCapture(path)
    except:
        return False

    if not cap.isOpened():
        return False

    try:
        video_reader = VideoReader(path, num_threads=1)
        return True
    except:
        return False

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(video_path):
    subtitle_path = video_path.replace('videos','subtitles').replace('.mp4','.srt')
    if os.path.exists(subtitle_path):
        subtitles = {}
        with open(subtitle_path, "r", encoding="utf-8") as file:
            content = file.read().split("\n\n")
            for section in content:
                if section.strip():
                    lines = section.split("\n")
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        subtitles[(start_time, end_time)] = text
    else:
        subtitle_path = video_path.replace('videos','subtitles').replace('.mp4','_en.json')
        data_li = json.load(open(subtitle_path))
        subtitles = {}
        for dic in data_li:
            start_time = parse_subtitle_time(dic["start"])
            end_time = parse_subtitle_time(dic["end"])
            subtitles[(start_time, end_time)] = dic['line']
    
    return subtitles

def extract_subtitles(video_path):
    subtitles = load_subtitles(video_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        pattern = r'<font color="white" size=".72c">(.*?)</font>'
        raw_text = re.findall(pattern, text)
        try:
            text = raw_text[0]
            subtitle_frames.append((float(start_time), float(end_time), text))
        except:
            subtitle_frames.append((float(start_time), float(end_time), text))

    return subtitle_frames

class Retrieval_Manager():
    def __init__(self, args=None, batch_size=1, clip_save_folder=None, clip_duration=30):
        
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        if args.retriever_type=='large':
            path = 'LanguageBind/LanguageBind_Video_FT'
        elif args.retriever_type=='huge':
            path = 'LanguageBind/LanguageBind_Video_Huge_V1.5_FT'
        else:
            raise KeyError

        clip_type = {
            'video':  path, # also LanguageBind_Video
            'image': 'LanguageBind/LanguageBind_Image'
        }

        self.model = LanguageBind(clip_type=clip_type, cache_dir='./model_zoo')

        self.text_retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


        self.model.eval()

        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(path)
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}

        self.clip_embs_cache = {}
        self.frame_embs_cache = {}
        self.batch_size = 1
        self.clip_save_folder = clip_save_folder
        self.args=args

        self.load_model_to_cpu()


    def load_model_to_device(self, device):

        self.model.to(device)

        def recursive_to(module):
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.nn.Module):
                    attr.to(device)
                    recursive_to(attr)
                elif isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))
                elif isinstance(attr, (list, tuple)):
                    new_attrs = []
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            item.to(device)
                            recursive_to(item)
                        elif isinstance(item, torch.Tensor):
                            item = item.to(device)
                        new_attrs.append(item)
                    setattr(module, name, type(attr)(new_attrs))

        recursive_to(self.model)

    def load_model_to_cpu(self):
        self.device=torch.device('cpu')
        self.load_model_to_device(torch.device('cpu'))
    
    def load_model_to_gpu(self, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.load_model_to_device(torch.device(f'cuda:{gpu_id}'))

    def cut_video(self, video_path, clip_save_folder=None, total_duration=-1):
        valid_clip_paths = set()
        time1 = time.time()
        os.makedirs(clip_save_folder, exist_ok=True)

        duration = VideoFileClip(video_path).duration
        chunk_number = math.ceil(duration/self.args.clip_duration) #! args.clip_duration= 10

        total_video_clip_paths = []
        for i in range(chunk_number):
            start_time = self.args.clip_duration * i
            end_time = start_time + self.args.clip_duration
            output_filename = f'clip_{i}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4'  
            total_video_clip_paths.append(clip_save_folder+'/'+output_filename)     

        if os.path.exists(clip_save_folder):
            retry = 0
            while retry < 2:
                valid_clip_num = 0
                for clip_name in os.listdir(clip_save_folder):
                    try:
                        VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                        valid_clip_paths.add(clip_save_folder+'/'+clip_name)
                        valid_clip_num+=1
                        del total_video_clip_paths[total_video_clip_paths.index(clip_save_folder+'/'+clip_name)]
                    except Exception as e:
                        print(f'clip {clip_name} is not valid, removing it: {e}')
                        
                if valid_clip_num >= (2*chunk_number//3): 
                    return [file for file in sorted(valid_clip_paths, key=lambda x: int(x.split('/')[-1].split('_')[1]))]
                else:
                    assert False,f'valid_clip_num:{valid_clip_num} < chunk_number-3: {chunk_number-3}, clip_save_folder:{clip_save_folder}'

            # 5次之后移除所有不合法的clip
            for path in total_video_clip_paths:
                try:
                    VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                    valid_clip_num+=1
                except Exception as e:
                    os.system('rm -rf '+path)

        else:
            print(clip_save_folder,'no valid clips found, cutting video:', video_path)
        
        return sorted(list(valid_clip_paths), key=lambda x: int(x.split('/')[-1].split('_')[1]))

    def save_clip(self, clip, clip_save_folder, clip_index, start_time, end_time, fps):
        start_time_str = self.format_time(start_time)
        end_time_str = self.format_time(end_time)
        os.makedirs(clip_save_folder,exist_ok=True)
        clip_path = os.path.join(clip_save_folder, f"clip_{clip_index}_{start_time_str}_to_{end_time_str}.mp4")
        height, width, _ = clip[0].shape
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        for frame in clip:
            out.write(frame)

        out.release()
        return clip_path

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"

    def parse_time(self, time_str):
        hours, mins, secs = map(int, time_str.split('-'))
        total_seconds = hours * 3600 + mins * 60 + secs
        return total_seconds


    @ torch.no_grad()
    def calculate_video_clip_embedding(self, video_path, folder_path, total_duration=None):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]

        folder_path = f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}/'
        os.makedirs(folder_path,exist_ok=True)

        embedding_path = os.path.join(folder_path,video_name+'.pkl')
        clip_path = os.path.join(folder_path,video_name+'_clip_paths.pkl')

        if os.path.exists(embedding_path) and os.path.exists(clip_path):
            video_paths = pickle.load(open(clip_path,'rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))
        
            invalid_num, invalid_videos=0,[]
            for v in video_paths:
                if not is_valid_video(v):
                    invalid_num+=1
                    invalid_videos.append(v)

            if invalid_num<3:
                return video_paths, total_embeddings
            else:
                print(embedding_path,'exist but have not enough valid video number!!',invalid_videos[0])
        # print('calculating video embeddings')
        video_paths = self.cut_video(video_path, os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0]),total_duration) #! 返回valid video_clip paths

        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(video_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_video_paths = []
        for i in range(len(video_paths)):
            try:
                inputs = {'video': to_device(self.modality_transform['video'](video_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_video_paths.append(video_paths[i])
                    total_embeddings.append(embeddings['video'])
            except Exception as e:
                print(e)
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_video_paths,open(f'{folder_path}/{video_name}_clip_paths.pkl','wb'))
        return video_paths,total_embeddings




    def extract_frames(self, video_path, output_dir, fps=1):
        os.makedirs(output_dir, exist_ok=True)
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print(f"Failed to open {video_path}")
            return
        
        frame_rate = vid.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print(f"Failed to get FPS for {video_path}")
            return
        
        frame_interval = math.floor(frame_rate / fps)
        frame_idx = 0
        second = 0
        
        with tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), desc=os.path.basename(video_path)) as pbar:
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_filename = os.path.join(output_dir, f"frame_{second}.png")
                    cv2.imwrite(frame_filename, frame)
                    second += 1
                
                frame_idx += 1
                pbar.update(1)
        
        vid.release()

    @ torch.no_grad()
    def calculate_frame_embedding(self, video_path, folder_path, total_duration):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            os.makedirs(f'{self.args.dataset_folder}/embeddings/frame/{self.args.retriever_type}/',exist_ok=True)
            frame_paths = pickle.load(open(f'{folder_path}/{video_name}_frame_paths.pkl','rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))
            invalid_num=0
            for v in frame_paths:
                if not is_valid_video(v):
                    invalid_num+=1

            if invalid_num<5:
                return frame_paths,total_embeddings
            
        frame_folder = '/'.join(video_path.split('/')[:-2]) + '/dense_frames/' + video_path.split('/')[-1].split('.')[0] + '/'
        if not os.path.exists(frame_folder) or os.listdir(frame_folder)==[]:
            self.extract_frames(video_path, frame_folder, fps=1)
        frame_paths = [frame_folder + file for file in sorted(os.listdir(frame_folder),key = lambda x:float(x.split('/')[-1].split('_')[1].split('.')[0]))]

        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(frame_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_frame_paths = []
        for i in range(len(frame_paths)):
            try:
                inputs = {'image': to_device(self.modality_transform['image'](frame_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_frame_paths.append(frame_paths[i])
                    total_embeddings.append(embeddings['image'])
            except:
                pass
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_frame_paths,open(f'{folder_path}/{video_name}_frame_paths.pkl','wb'))
        return frame_paths,total_embeddings



    @ torch.no_grad()
    def calculate_video_embedding(self, video_path, folder_path):
        video_name = video_path.split('/')[-1].split('.')[0]
        os.makedirs(folder_path,exist_ok=True)
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            try:
                embedding = pickle.load(open(embedding_path,'rb'))
                return embedding
            except:
                pass

        try: 
            inputs = {'video': to_device(self.modality_transform['video'](video_path), self.device)}
            with torch.no_grad():
                embedding = self.model(inputs)
            pickle.dump(embedding,open(f'{folder_path}/{video_name}.pkl','wb'))
            return embedding
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()



    @ torch.no_grad()
    def calculate_text_embedding(self,text,video_path=None,flag_save_embedding=True):
        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                embeddings = pickle.load(open(embedding_path,'rb'))
                # print('use precalculated subtitle embeddings')
                return embeddings
            except:
                pass

        # print('calculating subtitle embeddings')
        inputs = {'language':to_device(self.tokenizer(text, max_length=77, padding='max_length',truncation=True, return_tensors='pt'), self.device)}

        with torch.no_grad():
            embeddings = self.model(inputs)
        if flag_save_embedding:
            pickle.dump(embeddings['language'],open(embedding_path,'wb'))
        torch.cuda.empty_cache()
        return embeddings['language']


    @ torch.no_grad()
    def calculate_subtitle_embedding(self,video_path,flag_save_embedding=False,merge_sentence=False):
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]
        subtitle_embs = self.calculate_text_embedding(subtitles,video_path,flag_save_embedding=True)
        subtitle_embs = subtitle_embs.cpu()
        return subtitles_with_time,subtitle_embs


    @ torch.no_grad()
    def get_informative_subtitles(self, query, video_path, top_k=1, total_duration=-1, return_embeddings=False,merge_sentence=False,flag_save_embedding=1):
        if not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','.srt')) and not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','_en.json')):
            return ''

        q_emb = self.text_retriever.encode(query, batch_size=12, max_length=256)['dense_vecs']
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]

        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                subtitle_embeddings = pickle.load(open(embedding_path,'rb'))
            except Exception as e:
                print(e)
                subtitle_embeddings = self.text_retriever.encode(subtitles, batch_size=12, max_length=256)['dense_vecs']
                if flag_save_embedding:
                    pickle.dump(subtitle_embeddings,open(embedding_path,'wb'))

        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()  # shape: (832,)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        return [subtitles_with_time[i] for i in top_k_indices]



    def subtitle2clips(self, subtitle_triple, video_path):
        def is_overlap(begin1, end1, begin2, end2):
            return begin1 <= end2 and begin2 <= end1

        subtitle_begin_time, subtitle_end_time = subtitle_triple[0], subtitle_triple[1]
        ans = []
        for clip in os.listdir(self.clip_save_folder + video_path.split('/')[-1][:-4]):
            clip_begin_time, clip_end_time = self.parse_time(clip.split('.')[0].split('_')[2]),self.parse_time(clip.split('.')[0].split('_')[4])
            if is_overlap(subtitle_begin_time, subtitle_end_time, clip_begin_time, clip_end_time):
                video_clip_path = self.clip_save_folder + video_path.split('/')[-1][:-4] +f'/{clip}'
                ans.append(video_clip_path)
        return ans

    @ torch.no_grad()
    def get_informative_clips_with_video_query(self,query, query_video_path,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        text_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        text_emb = text_emb / text_emb.norm(p=2, dim=1, keepdim=True)

        inputs = {'video': to_device(self.modality_transform['video'](query_video_path), self.device)}
        with torch.no_grad():
            q_emb = self.model(inputs)['video'].cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        q_emb = q_emb + text_emb

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        # Normalize the clip embeddings
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        return result



    @ torch.no_grad()
    def get_informative_clips(self,query,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        q_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        if video_path not in self.clip_embs_cache: #! self.clip_embs_cache dict[tuple[List, List]]
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        # Normalize the clip embeddings
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        return result



# todo: copy from prompt.py in vdr(https://github.com/yhy-2000/VideoDeepResearch#)
RETRIEVER_TOOL_PROMPT ='''You are a video understanding expert tasked with analyzing video content and answering single-choice questions. You will receive:  
- The total duration of the video (in seconds).  
- A question about the video.

## Available Tools  
You can call any combination of these tools in the same response, using one or more per step. Additionally, if you include multiple queries in the same call, they must be separated by ';'.

### 1. Video Segment Inquiry  
The video is segmented into segments of {clip_duration} seconds, and you can query them by their segment number `[0, ceil(total duration/{clip_duration})]`.  
You may also query multiple consecutive segments by concatenating their numbers (e.g., `112;113`).  

#### Query Formats:  
#### *Single Segment Query*  
```<video_reader>segment_number</video_reader><video_reader_question>your_question</video_reader_question>```  

#### *Sequential Segments Query*  
```<video_reader>segment_N;segment_N+1</video_reader><video_reader_question>your_question</video_reader_question>```  

- Rules:  
  - Only *temporally adjacent segments* supported, so you must first order all segments FROM SMALLEST TO LARGEST and then concatenate those that are adjacent in time. (e.g. N;N+1 are temporally adjacent segments, but N;N+2 are not.)
  - Max 2 segments per query* (split longer sequences into multiple 2-segment batches).  

- Use case:
  - Allows asking questions about the video segments returned by the retriever.
  - If the question mentions a specific timestamp, it can target the corresponding video segment.
    For example:
      - "What happens at 00:15?" -> Query `<video_reader>1</video_reader>`
      - "Describe the action in the first minute." -> Query `<video_reader>0;1</video_reader> <video_reader>2;3</video_reader> <video_reader>4;5</video_reader>`

- Important Notes:
  - You should question about every retrieved video segment without any omission!!!!! 
  - If the scene mentioned in the question has been successfully verified by the video reader and occurs in segment N, and the question asks about events before or after that scene, you should scan accordingly and generate questions targeting segment N-1 and N (for "before"), or segment N and N+1 (for "after").
  - For counting/order problems, the question should follow this format "For questions asking whether a specific action occurs, you should carefully examine each frame — if even a single frame contains the relevant action, it should be considered as having occurred. The question is: is there xxx?"
  - For anomaly detection, don't concate segments and raise single segment query.
  - For anomaly detection, provide all the candidate options in each question!!
  - For anomaly detection, you may concatenate up to 10 sequential video segments, including the retrieved segment and its neighboring segments, to obtain a comprehensive overview of the event.
  - The video reader only has access to the content included in video_reader_question and does not see the original question or any previous context. Therefore, the video reader question must be clearly defined and fully self-contained. Avoid any ambiguous references such as "Same question as above." Each question should stand on its own!!!

### 2. Video Segment Retrieval  
```<video_segment_retriever_textual_query>search_query</video_segment_retriever_textual_query><topk>topk</topk>```

- Use case:
  - Returns a ranked list of segments, with the most relevant results at the top. For example, given the list [d, g, a, e], segment d is the most relevant, followed by g, and so on.
  - Assign topk=15 for counting problem, assign lower topk=8 for other problem
  - Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries). 
  - The video_segment_retriever may make mistakes, and the video_reader is more accurate than the retriever. If the retriever retrieves a segment but the video_reader indicates that the segment is irrelevant to the current query, the result from the video_reader should be trusted.
  - Each time the user returns retrieval results, you should query all the retrieved segments in the next round. If clips retrieved by different queries overlap, you can merge all the queries into a single question and access the overlapping segment only once using video_reader.
  - You must generate a question for **every segment returned by the video retriever** — do not miss any one!!!!!
  - The video retriever cannot handle queries related to the global video timeline, because the original temporal signal is lost after all video segments are split. If a question involves specific video timing, you need to boldly hypothesize the possible time range and then carefully verify each candidate segment to locate the correct answer.


### 3. Video Browsing
<video_browser_question>your_question</video_browser_question>```  

- Use case:
  - For holistic question that are not specific to a segment, such as "What is the video mainly about?", "What genre is this video?", you can use the video_browser to coarsely browse the video and answer the question.

## Execution Process  

### Step 1: Analyze & Think  
- Document reasoning in `<thinking></thinking>`.  
- Output one or more tool calls (strict XML format).  
- Stop immediately after and output `[Pause]` to wait for results.  
- Don't output anything after [Pause] !!!!!!!

### Step 2: Repeat or Answer  
- If more data is needed, repeat Step 1 until you could provide an accurate answer. The maximum number of iterations is {MAX_DS_ROUND}.
- If ready, output:  
  ```<thinking>Final reasoning</thinking><answer>(only the letter (A, B, C, D, E, F, ...) of the correct option)</answer>```  
---

## Suggestions  
1. Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries). 
2. For counting problems, consider using a higher top-k and more diverse queries to ensure no missing items. 
3. To save the calling budget, it is suggested that you include as many tool calls as possible in the same response, but you can only concatenate video segments that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of two at a time!!
4. Suppose the `<video_segment_retriever_textual_query>event</video_segment_retriever_textual_query>` returns a list of segments [a,b,c,d,e]. If the `video_reader` checks each segment in turn and finds that none contain the event, but you still need to locate the segment where the event occurs, then by default, assume the event occurs in the top-1 segment a.**
5. For counting problems, your question should follow this format: Is there xxx occur in this video? (A segment should be considered correct as long as the queried event occurs in more than one frame, even if the segment also includes other content or is primarily focused on something else. coarsely matched segments should be taken into account (e.g., watering flowers vs. watering toy flowers)) 
6. For counting problems, you should carefully examine each segment to avoid any omissions!!!
7. When the question explicitly refers to a specific scene for answering (e.g., "What did Player 10 do after scoring the first goal?"), you must first use the video retriever and video reader to precisely locate that scene. Once the key scene is identified—e.g., the moment of Player 10's first goal in segment N—you should then generate follow-up questions based only on that segment and its adjacent segments. For example, to answer what happened after the first goal, you should ask questions targeting segment N and segment N+1.
8. Don't output anything after [Pause] !!!!!!!

## Strict Rules  
1. Response of each round should provide thinking process in <thinking></thinking> at the beginning!! Never output anything after [Pause]!!
2. You can only concatenate video segments that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of TWO at a time!!!
3. If you are unable to give a precise answer or you are not sure, continue calling tools for more information; if the maximum number of attempts has been reached and you are still unsure, choose the most likely one.
4. You must generate a question for **every segment returned by the video retriever** — do not miss any one!!!!!
5. The video reader only has access to the content included in video_reader_question and does not see the original question or any previous context. Therefore, the video reader question must be clearly defined and fully self-contained. Avoid any ambiguous references such as "Same question as above." Each question should stand on its own!!!
6. Never guess the answer, question about every choice, question about every segment retrieved by the video_segment_retriever!!!! 
---

### Input  
Question: {question}
Video Duration: {duration} seconds
(Never assuming anything!!! You must rigorously follow the format and call tools as needed. Never assume tool results!! Instead, think, call tools, output [Pause] and wait for the user to supply the results!!! Don't output anything after [Pause] !!!!!)'''