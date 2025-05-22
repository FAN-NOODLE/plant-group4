import cv2

def test_cameras():
    # Test the first 5 camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is available")
                # Display camera feed
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(1000)  # Display for 1 second
            else:
                print(f"Camera {i} cannot read frame")
        else:
            print(f"Camera {i} is not available")
        cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_cameras() 