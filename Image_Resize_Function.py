import cv2


def image_resize(img, w, h):

    img = img

    # Resize the image to fit within the screen dimensions
    screen_width = w  # Set your desired width
    screen_height = h  # Set your desired height

    # Calculate the aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]  # width / height

    # Resize while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
        print(new_height)
    else:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)
        print(new_width)

    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

    # cv2.imshow("Image", resized_img)
    # cv2.waitKey(0)  # 0 means wait indefinitely
    # cv2.destroyAllWindows()

    # Save the image to the same folder
    # cv2.imwrite("Sample_Pictures_And_Videos/Hamdan.jpg", resized_img)
