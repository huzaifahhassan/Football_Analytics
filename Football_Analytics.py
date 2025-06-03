# Importing Libraries
import comet_ml
import os
import roboflow
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
from Image_Resize_Function import image_resize
import supervision as sv
from importlib import reload
from pathlib import Path
import gc
import umap
from sklearn.cluster import KMeans
import numpy as np
from more_itertools import chunked
import torch
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
from inference import get_model
from sports.sports.annotators import soccer
from sports.sports.configs.soccer import SoccerPitchConfiguration


#====================================================================================================
from football_functions import init_team_classifier
from football_functions import resolve_goalkeepers_team_id
from football_functions import set_model_transformer
from football_functions import get_draw_pitch_config

#set directory to current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir = script_dir # Change the current working directory

#os.chdir = "C:/Users/huzin/Downloads/Huzaifah_Folder/Computer_Vision_Course/Football_D_S_T/sports"
os.chdir = os.path.join(script_dir, "sports")
from sports.sports.annotators import soccer
os.chdir = script_dir # Change the current working directory
#====================================================================================================
CONFIG = get_draw_pitch_config()
#=====================================================================================================

#=====================================================================================================

# Open the video file

def football_analytics(video_name):

    team_classifier = init_team_classifier(video_name=video_name)

    video_mini_path = os.path.join("Videos", video_name)
    video_path = os.path.join(script_dir, video_mini_path)

    model = YOLO("runs/detect/train2/weights/best.pt")

    # Adding Tracker
    tracker = sv.ByteTrack(track_activation_threshold=0.25)
    tracker.reset()

    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1493"), radius=200
    )

    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.ColorPalette.from_matplotlib('viridis', 2),
        radius=200,
    )

    vertex_annotator_3 = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1800"),
        radius=200,
    )

    cap = cv2.VideoCapture(video_path)

    # Initialize centroids for Team A and Team B
    prev_centroids = None

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame and get the results
            results = model.predict(
                source=frame,
                conf=0.5,
                device = "cuda:0"
            )

            # Getting the detections from the results
            detections = sv.Detections.from_ultralytics(results[0])

            # Ellipse Annotator Instance
            ellipse_annotator = sv.EllipseAnnotator(
                color=sv.ColorPalette.from_hex(
                    [
                        "#FF0000",  # colour is red
                        "#00FF00",  # colour is green
                        "#0000FF",  # colour is blue
                    ]
                )
            )

            # Triangle Annotator Instance
            triangle_annotator = sv.TriangleAnnotator(
                base=20,
                height=20,
                color=sv.Color.BLUE,
            )

            # Creating the Label Annotator Instance

            from supervision.geometry.core import Position

            label_annotator = sv.LabelAnnotator(
                text_color=sv.Color.BLACK,
                text_padding=2,
                text_position=Position.TOP_LEFT,
                text_scale=0.4,
            )

            # Filtering Ball Detections and Retaining just Player, Goal Keeper and Referee Detections
            player_ref_goalkeeper_detections = detections[detections.class_id != 0]
            player_ref_goalkeeper_detections = player_ref_goalkeeper_detections.with_nms(
                threshold=0.3, class_agnostic=True
            )

            # Applying Tracking to the Detections
            player_ref_goalkeeper_detections = tracker.update_with_detections(
                detections=player_ref_goalkeeper_detections
            )

            # getting just player detections
            player_detections = player_ref_goalkeeper_detections[
                player_ref_goalkeeper_detections.data["class_name"] == "player"
            ]

            # getting just ball detections
            ball_detections = detections[detections.data["class_name"] == "ball"]

            # Getting the Crops
            crops = []
            for i, xyxy in enumerate(player_detections.xyxy):
                crop = sv.crop_image(frame, xyxy)
                id = player_ref_goalkeeper_detections.class_id[i]
                label = player_ref_goalkeeper_detections.data["class_name"][i]
                crops.append(crop)
                # resize the annotated image for display
                crop = image_resize(crop, w=600, h=600)

            cluster_labels = team_classifier.predict(crops)  # 0 for Team A, 1 for Team B
            cluster_labels = np.where(cluster_labels == 0, 2, cluster_labels).astype(
                int
            )  # removing the 0 labels

            # getting just the goal keeper detections
            goal_keeper_detections = player_ref_goalkeeper_detections[
                player_ref_goalkeeper_detections.data["class_name"] == "goalkeeper"
            ]

            # replacing player IDs with Team IDs (cluster labels)
            player_detections.class_id = cluster_labels

            # now we will assign IDs (1 or 2) to the goal keepers
            gk_team_id = resolve_goalkeepers_team_id(
                player_detections, goal_keeper_detections
            )

            #***********************************************************

            # replacing gk IDs with calculated gk IDs
            goal_keeper_detections.class_id = gk_team_id.astype(int)

            referees_detections = player_ref_goalkeeper_detections[
                player_ref_goalkeeper_detections.data["class_name"] == "referee"
            ]

            all_detections = sv.Detections.merge(
                [player_detections, goal_keeper_detections, referees_detections]
            )

            # Transforming the points to the frame using the model transformer
            custom_transformer , real_pitch_points = set_model_transformer(frame) # transformer is initialised
            # getting player points
            player_points = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            # getting goal keeper points
            goal_keeper_points = goal_keeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            # getting referee points
            referee_points = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            # getting ball points
            ball_points = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            # Annotating the points on the frame
            # top_annotated_frame = top_view_annotated_frame.copy()

            top_annotated_frame = soccer.draw_pitch(CONFIG , scale = 1)

            # applying transformation to the points
            if player_points.shape[0] >= 1:
                trans_player_points = custom_transformer.transform_points(points=player_points)
                player_key_points = sv.KeyPoints(xy=trans_player_points[np.newaxis, ...])
                player_key_points.class_id = player_detections.class_id

                print("Num of Keypoints" , len(player_key_points.xy))

                for i in range(len(player_key_points.xy[0])):
                    if player_key_points.class_id[i] == 1:
                        cv2.circle(top_annotated_frame, tuple(player_key_points.xy[0,i,:].astype('int32')), 200, (255,255,0), -1)
                    else:
                        cv2.circle(top_annotated_frame, tuple(player_key_points.xy[0,i,:].astype('int32')), 200, (255,0,0), -1)
        
            if goal_keeper_points.shape[0] >= 1:
                trans_goal_keeper_points = custom_transformer.transform_points(points=goal_keeper_points)
                goal_keeper_key_points = sv.KeyPoints(xy=trans_goal_keeper_points[np.newaxis, ...])
                # top_annotated_frame = vertex_annotator.annotate(
                #     scene=top_annotated_frame, key_points=goal_keeper_key_points
                # )
            if referee_points.shape[0] >= 1:
                trans_referee_points = custom_transformer.transform_points(points=referee_points)
                referee_key_points = sv.KeyPoints(xy=trans_referee_points[np.newaxis, ...])
                referee_key_points.class_id = referees_detections.class_id
                # top_annotated_frame = vertex_annotator_3.annotate(
                #     scene=top_annotated_frame, key_points=referee_key_points
                # )
            if ball_points.shape[0] >= 1:
                trans_ball_points = custom_transformer.transform_points(points=ball_points)
                ball_key_points = sv.KeyPoints(xy=trans_ball_points[np.newaxis, ...])
                top_annotated_frame = vertex_annotator.annotate(
                    scene=top_annotated_frame, key_points=ball_key_points
                )

            # Applying Ellipse Annotations using Ellipse Annotator Instance
            annotated_frame = ellipse_annotator.annotate(frame, detections=all_detections)

            # Getting the Ball Detections and Padding the Boxes
            ball_detections = detections[detections.class_id == 0]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # Applying Triangle Annotations using Triangle Annotator Instance
            annotated_frame = triangle_annotator.annotate(
                annotated_frame, detections=ball_detections
            )

            # Applying Label Annotations using Label Annotator Instance

            labels = [
                # f"{class_name} {confidence:.2f}"
                # for class_name, confidence in zip(detections["class_name"], detections.confidence)
                f"#{tracker_id}"
                for tracker_id in all_detections.tracker_id
            ]

            annotated_frame = label_annotator.annotate(
                annotated_frame, detections=all_detections, labels=labels
            )

            # resize the annotated image for display
            annotated_image = image_resize(annotated_frame, w=800, h=800)
            top_annotated_frame = image_resize(top_annotated_frame, w=800, h=800)

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_image)
            cv2.imshow("Top View", top_annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return None

### Running the Function
#football_analytics(video_name=video_name)