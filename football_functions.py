from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import cv2
from Image_Resize_Function import image_resize
import numpy as np
import os
from inference import get_model
from sports.sports.annotators import soccer
from sports.sports.configs.soccer import SoccerPitchConfiguration

#set directory to current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir = script_dir # Change the current working directory

#====================================================================================================

#====================================================================================================
def get_crops(video_path , model_path , stride = 3000):

    # Open the video file
    # video_path = video_path # This line is redundant as video_path is already a parameter

    model = YOLO(model_path)

    # No of frames to skip
    # stride = stride # This line is redundant as stride is already a parameter

    # Create a video frames generator
    frame_generator = sv.get_video_frames_generator(video_path , stride=stride)

    crops = []

    # Loop through the video frames
    for frame in tqdm(frame_generator,desc = "Frame_Iteration"):
        # Run YOLO inference on the frame and get the results
        results = model.predict(
            source=frame,
            conf=0.5,
        )

        # Getting the detections from the results
        detections = sv.Detections.from_ultralytics(results[0])
        # The following line seems to be for debugging or a specific use case,
        # it's not directly used for the final 'player_ref_goalkeeper_detections'
        player_detections = detections[detections.data["class_name"] == "player"]
        print(player_detections)


        # Filtering Ball Detections and Retaining just Player, Goal Keeper and Referee Detections
        # Assuming class_id 0 is for 'ball'
        player_ref_goalkeeper_detections = detections[detections.class_id != 0]
        player_ref_goalkeeper_detections = player_ref_goalkeeper_detections.with_nms(
            threshold=0.3, class_agnostic=True
        )
        for i , xyxy in enumerate(player_ref_goalkeeper_detections.xyxy):
            crop = sv.crop_image(frame, xyxy)
            # id = player_ref_goalkeeper_detections.class_id[i] # 'id' is assigned but not used
            # label = player_ref_goalkeeper_detections.data["class_name"][i] # 'label' is assigned but not used forimshow
            crops.append(crop)

            # resize the annotated image for display
            # The resized crop is not stored back or used further in this loop iteration's logic for 'crops' list
            resized_display_crop = image_resize(crop, w=600, h=600)
            # cv2.imshow(f"Object Type: {player_ref_goalkeeper_detections.data['class_name'][i]}", resized_display_crop)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # print(crops)
        # print(player_ref_goalkeeper_detections)

    return crops

def process_video_and_display_crops():
    """
    Sets up paths and parameters, calls get_crops, and plots the results.
    """
    #video_path = "C:/Users/huzin/Downloads/Huzaifah_Folder/Computer_Vision_Course/Football_D_S_T/Videos/121364_0.mp4"
    video_path = os.path.join(script_dir, "Videos/121364_0.mp4")
    model_path = "runs/detect/train2/weights/best.pt"
    stride = 2000

    extracted_crops = get_crops(video_path , model_path , stride)
    
    if extracted_crops: # Check if crops list is not empty before plotting
        # Determine grid size dynamically or cap it if too many crops
        num_crops = len(extracted_crops)
        # For demonstration, let's stick to the original 10x10 or adjust if fewer crops.
        # This part can be made more robust.
        grid_cols = 10
        grid_rows = (num_crops + grid_cols - 1) // grid_cols # Calculate rows needed
        if num_crops == 0:
            print("No crops were extracted.")
            return
        elif num_crops < 100: # If less than 100 crops, adjust grid size
             sv.plot_images_grid(extracted_crops[:], grid_size=(grid_rows, grid_cols))
        else: # If 100 or more crops, use the original 10x10 grid for the first 100
             sv.plot_images_grid(extracted_crops[:100], grid_size=(10, 10)) # Display up to 100 crops
    else:
        print("No crops were extracted to display.")

# To run the processing:
if __name__ == "__main__":
    # Ensure 'Image_Resize_Function.py' exists and 'image_resize' is importable.
    # Also, ensure the video and model paths are correct.
    process_video_and_display_crops()


#====================================================================================================

#====================================================================================================
def resolve_goalkeepers_team_id(
    players: sv.Detections, goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    #print(len(players.class_id))
    #print(len(players[players.class_id == 2]))
    #print(len(players[players.class_id == 1]))
    #print(len(goalkeepers_xy))
    team_2_centroid = players_xy[players.class_id == 2].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_2 = np.linalg.norm(goalkeeper_xy - team_2_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(2 if dist_2 < dist_1 else 1)

    return np.array(goalkeepers_team_id)
#====================================================================================================

#====================================================================================================
### Team Classifier from Supervision
def init_team_classifier(video_name):

    os.chdir = script_dir # Change the current working directory

    model = YOLO("yolo11l.pt")  # load a pretrained YOLOv8n model

    from sports.sports.common.team import TeamClassifier

    #SOURCE_VIDEO_PATH = "C:/Users/huzin/Downloads/Huzaifah_Folder/Computer_Vision_Course/Football_D_S_T/Videos/2e57b9_0.mp4"
    SOURCE_VIDEO_PATH = os.path.join(script_dir, "Videos" , video_name )

    PLAYER_ID = 2
    STRIDE = 30

    frame_generator = sv.get_video_frames_generator(
        source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

    crops = []
    for frame in frame_generator:
        result = model.predict(source=frame,conf=0.3,device = "cuda:0")[0]
        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)

    return team_classifier
#====================================================================================================

#====================================================================================================
### Pitch Keypoint Detection and Transformation

### Getting Keypoint Detection Model
# load a pre-trained yolov8n model
pitch_keypoint_model = get_model(
    model_id="football-field-detection-f07vi/15", api_key="022GaTg2BBRBT6lOHMzJ"
)

#====================================================================================================

def get_draw_pitch_config():
    # set current working directory
    os.chdir = script_dir # Change the current working directory


    CONFIG = SoccerPitchConfiguration()
    top_view_annotated_frame = soccer.draw_pitch(CONFIG , scale=1)
    #print(top_view_annotated_frame.shape)
    #sv.plot_image(top_view_annotated_frame)
    return CONFIG


### Defining View Transformer Class
class ViewTransformer:
    # Finding the Rotation Matrix
    def __init__(self, source:np.ndarray, target:np.ndarray):
        source = source.astype(np.float32) # not accessible , just used to caliculate homography
        target = target.astype(np.float32) # not accessible , just used to caliculate homography
        self.m , _ = cv2.findHomography(source, target)

    # Transforming the Persoective points to Top View Points using Rot Matrix
    def transform_points(self, points:np.ndarray) -> np.ndarray:
        points = points.reshape(-1,1,2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1,2).astype(np.float32)
    
    
def set_model_transformer(frame):  

    CONFIG = get_draw_pitch_config()

    result = pitch_keypoint_model.infer(frame, confidence=0.3)[0]

    key_points = sv.KeyPoints.from_inference(result)
    filter = key_points.confidence[0] > 0.5
    points = key_points.xy[0][filter]

    # producing keypoints object
    frame_reference_key_points = sv.KeyPoints(xy=points[np.newaxis, ...])

    pitch_reference_points = np.array(CONFIG.vertices)[
        filter
    ]  # This filter filters the vertices as well

    model_transformer = ViewTransformer(
        source=frame_reference_key_points.xy[0].reshape(-1, 2),
        target=pitch_reference_points.reshape(-1, 2)
    )

    return (model_transformer , points)
