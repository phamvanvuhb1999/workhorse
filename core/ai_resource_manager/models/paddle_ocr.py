import copy

import os

import ml2rt
import numpy as np
import cv2 as cv

from configs.common import BASE_DIR
from core.ai_resource_manager import RedisAIModel
from core.ai_resource_manager.models.utils.postprocess import get_boxes_from_bitmap, filter_tag_det_res, sorted_boxes, \
    get_rotate_crop_image
from core.ai_resource_manager.models.utils.preprocess import resize_image, normalize_image, cvt_hwc_to_chw, \
    get_data_by_keys


class PaddleDetectorRedisModel(RedisAIModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        preprocess = kwargs.get("preprocess", {})
        postprocess = kwargs.get("postprocess", {})
        input_tensor_name = kwargs.get("input_tensor_name", "x")
        output_tensor_name = kwargs.get("output_tensor_name", "save_infer_model/scale_0.tmp_0")

        self.model_key = "paddle_detect"

        self.input_tensor_name: str = input_tensor_name
        self.output_tensor_name: str = output_tensor_name
        self.preprocess_arg = {
            "limit_side_len": preprocess.get("limit_side_len", 960),
            "scale": preprocess.get("scale", 1.0 / 255.0),
            "mean": preprocess.get("mean", (0.485, 0.456, 0.406)),
            "std": preprocess.get("std", (0.229, 0.224, 0.225)),
            "keep_keys": preprocess.get("keep_keys", ["image", "shape"]),
        }

        self.postprocess_arg = {
            "thresh": postprocess.get("thresh", 0.3),
            "max_candidates": postprocess.get("max_candidates", 1000),
            "min_size": postprocess.get("min_size", 3),
            "box_thresh": postprocess.get("box_thresh", 0.6),
            "unclip_ratio": postprocess.get("unclip_ratio", 1.5),
        }


    def initiate(
        self,
        backend: str = "onnx",
        device: str = "cpu",
        model_path: str = "",
    ):
        if not model_path:
            model_path = os.path.join(BASE_DIR, "assets/text_detection/PP_OCR_v3_det.onnx")

        paddle_det = ml2rt.load_model(model_path)

        self.store_model(key=self.model_key, backend=backend, device=device, data=paddle_det)

    def preprocess(self, data: dict):
        data = resize_image(data, self.preprocess_arg["limit_side_len"])
        data = normalize_image(
            data,
            self.preprocess_arg["scale"],
            self.preprocess_arg["mean"],
            self.preprocess_arg["std"],
        )
        data = cvt_hwc_to_chw(data)
        data_list = get_data_by_keys(data, self.preprocess_arg["keep_keys"])
        return data_list

    def postprocess(
        self,
        preds: np.ndarray,
        shape_list: np.ndarray,
    ):
        pred = preds[:, 0, :, :]
        segmentation = pred > self.postprocess_arg["thresh"]
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = get_boxes_from_bitmap(
                pred[batch_index],
                mask,
                src_w,
                src_h,
                self.postprocess_arg["max_candidates"],
                self.postprocess_arg["min_size"],
                self.postprocess_arg["box_thresh"],
                self.postprocess_arg["unclip_ratio"],
            )
            boxes_batch.append({"points": boxes})
        return boxes_batch

    def process(self, image: np.ndarray | bytes, **kwargs):
        from core.queueing.tasks import push_to_source_manager
        from core.ai_resource_manager import SVTRCLNetRecognizerRedisModel

        if isinstance(image, bytes):
            image_array = np.frombuffer(image, np.uint8)
            image = cv.imdecode(image_array, cv.IMREAD_COLOR)

        cp_img = image.copy()
        data: dict = {
            "image": cp_img
        }

        img, shape_list  = self.preprocess(data)
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        self.feed_model(key=self.input_tensor_name, tensor=img)

        # Execute model
        self.execute_model(key=self.model_key, inputs=[self.input_tensor_name,], outputs=[self.output_tensor_name, ])

        # Get output tensor
        outputs = self.get_tensor_model(key=self.output_tensor_name)
        # Should release model lock after get output tensor
        self.release_model_lock(**kwargs)

        if outputs is None:
            return

        post_result = self.postprocess(preds=outputs, shape_list=shape_list)

        dt_boxes = post_result[0]["points"]

        dt_boxes = filter_tag_det_res(dt_boxes, image.shape)

        dt_boxes = sorted_boxes(dt_boxes)

        img_crop_list = [
            get_rotate_crop_image(image, copy.deepcopy(dt_box))
            for dt_box in dt_boxes
        ]

        # Push new rec task to queue
        task_kwargs = {
            **kwargs,
            "model_type": SVTRCLNetRecognizerRedisModel.class_prefix(),
            "images": img_crop_list,
            "dt_boxes": dt_boxes,
        }
        push_to_source_manager.apply_async(
            kwargs=task_kwargs
        )
        return {
            "step": self.class_prefix(),
            "is_finished": False
        }


if __name__ == "__main__":
    from redis import Redis
    from datetime import datetime

    # paddle_det = PaddleDetectorRedisModel.get_instance(prefix=f"{PaddleDetectorRedisModel.class_prefix()}:bb956221-c326-4d2b-a41a-7c4ae7a3ec5d")
    # redis_cli = Redis()
    # if not redis_cli.keys("paddledetectorredismodel:*"):
    #     paddle_det.initiate()
    # imgg = cv.imread(r"/home/vupham/Downloads/WhatsApp Image 2024-12-03 at 15.07.55.jpeg", cv.IMREAD_GRAYSCALE)
    # imgg = cv.cvtColor(imgg, cv.COLOR_GRAY2BGR)
    # # with open(r"/home/vupham/Downloads/WhatsApp Image 2024-12-03 at 15.07.55.jpeg", "rb") as f:
    # #     data = f.read()
    # #     image_array = np.frombuffer(data, np.uint8)
    # #     image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    # start_time = datetime.now()
    # print(f"start time {start_time}")
    # result = paddle_det.process(imgg)
    # print(f"end time: {datetime.now()}, executed time {datetime.now() - start_time}")
    # # print(result)

    paddle_det = PaddleDetectorRedisModel.get_instance()
    paddle_det.initiate()