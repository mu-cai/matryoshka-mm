"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import uuid
import math
import pandas as pd

from tqdm import tqdm

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_processor.llava2_model_processor import *
from vision_processor.fps_gridview_processor import *
from .record import *

def split_dataframe(df, n):
    """Split a DataFrame into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(df) / n)
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def get_dataframe_chunk(df, n, k):
    """Retrieve the k-th chunk from the DataFrame split into n chunks"""
    chunks = split_dataframe(df, n)
    if k < len(chunks):
        return chunks[k]
    else:
        raise IndexError("Chunk index out of range")
    
    
    
class LlavaPipeline:
    def __init__(
        self,
        model_name,
        path_qa,
        path_video_file_format,
        dir="./llava_pipeline_result/",
        matryoshka_vis_token_scale= None,
        args = None,
    ):
        self.model_name = "mucai/" + model_name
        self.path_qa = path_qa
        self.path_dir = dir
        self.path_result = dir
        self.path_video_file_format = path_video_file_format
        self.error_video_name = []
        self.num_chunks = getattr(args, 'num_chunks', 1)
        self.chunk_idx = getattr(args, 'chunk_idx', 0)
        self.make_video_file_list()
        self.load_model(matryoshka_vis_token_scale=matryoshka_vis_token_scale)

    def make_video_file_list(self):
        self._load_qa_file()
        self.df_qa["path_video"] = self.df_qa.apply(
            lambda x: (self.path_video_file_format % (x["video_name"])), axis=1
        )

    def load_model(self, matryoshka_vis_token_scale = None):
        self.model = Llava2Processor(self.model_name)
        self.model.load_model(matryoshka_vis_token_scale = matryoshka_vis_token_scale)

    def set_component(
        self,
        user_prompt,
        frame_fixed_number=6,
        func_user_prompt=lambda prompt, row: prompt % (row["question"]),
        calculate_max_row=lambda x: round(math.sqrt(x)),
    ):

        if not hasattr(self, "model"):
            raise AttributeError("Model is not loaded. Please call load_model()")

        self.frame_fixed_number = frame_fixed_number
        self.user_prompt = user_prompt
        self.func_user_prompt = func_user_prompt
        self.calculate_max_row = calculate_max_row

        self.fps_data_processor = FpsDataProcessor(
            save_option=SaveOption.IMAGE,
            calcualte_max_row=self.calculate_max_row,
            frame_fixed_number=self.frame_fixed_number,
        )

        extra_dir = "ffn=%s/" % (str(self.frame_fixed_number))
        self._make_directory(extra_dir)

    def do_pipeline(self):
        print("start pipeline")

        for idx, row in tqdm(self.df_qa.iterrows()):
            question_id = str(row["question_id"])
            video_path = row["path_video"]
            ts = row["ts"] if "ts" in row else None
            video_extensions = ["avi", "mp4", "mkv", "webm", "gif"]

            if not os.path.exists(video_path):
                if os.path.exists(video_path.replace('/v_', '/')):
                    video_path = video_path.replace('/v_', '/')
                base_video_path, _ = os.path.splitext(video_path)
                for ext in video_extensions:
                    temp_path = f"{base_video_path}.{ext}"
                    if os.path.exists(temp_path):
                        video_path = temp_path
                        break

            if not os.path.exists(self._make_file_path(question_id)):
                try:
                    image_data = self.fps_data_processor.process([video_path], ts)

                    answer = self.model.infer_and_save(
                        user_prompt=self.func_user_prompt(self.user_prompt, row),
                        raw_image=image_data,
                    )
                    if -1 != answer:
                        self.write_result_file(question_id, answer)
                    else:
                        self.error_video_name.append(video_path)
                except Exception as e:
                    print(e)
                    print(video_path)
                    continue

        return self.merge_qa_and_answer()

    def write_result_file(self, question_id, answer, extension=".txt"):
        file_path = self._make_file_path(question_id, extension)
        with open(file_path, "w") as file:
            file.write(answer)

    def _make_file_path(self, question_id, extension=".txt"):
        return os.path.join(self.path_result, question_id + extension)

    def _load_qa_file(self):
        try:
            self.df_qa = pd.read_csv(self.path_qa, index_col=0)
            if self.num_chunks != 1:
                self.df_qa = get_dataframe_chunk(self.df_qa, self.num_chunks, self.chunk_idx)
                print(f'Initialze Chunk {self.chunk_idx}')
        except Exception as e:
            print(e)
            raise Exception("not valid qa files")

    def _make_directory(self, extra_dir):
        self.path_result = os.path.join(self.path_dir, extra_dir)
        os.makedirs(self.path_result, exist_ok=True)

    def merge_qa_and_answer(self):
        print("start merge_qa_and_answer")

        self.df_qa["pred"] = None
        path_merged = os.path.join(self.path_result, "result.csv")

        if not os.path.exists(path_merged):
            for idx, row in self.df_qa.iterrows():
                question_id = str(row["question_id"])
                file_path = self._make_file_path(
                    question_id,
                )
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as file:
                            file_contents = file.read()
                        self.df_qa.loc[idx, "pred"] = file_contents
                    except Exception as e:
                        print(file_path)
                        raise (e)
                else:
                    print(f'path not found for {file_path}')

            self.df_qa.to_csv(path_merged)
        else:
            self.df_qa = pd.read_csv(path_merged, index_col=0)

        return self.df_qa, path_merged
