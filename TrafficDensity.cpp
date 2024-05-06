#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <vector>
#include <string>

int main() 
{
  // Threshold to detect objects
  float thres = 0.52;

  // Path to your video file
  std::string video_file_path = "demo.mp4";

  // Open video capture from video file
  cv::VideoCapture cap(video_file_path);

  // Set the resolution and frame rate (if needed)
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  cap.set(cv::CAP_PROP_FPS, 70);

  // Load class names from a text file
  std::vector < std::string > classNames;
  std::string classFile = "names";
  std::ifstream file(classFile);
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      classNames.push_back(line);
    }
    file.close();
  }

  // Load the pre-trained model (SSD = Single Shot Multibox Detector)
  std::string configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
  std::string weightsPath = "frozen_inference_graph.pb";

  // Initialize the model
  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(weightsPath, configPath);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // Create a window
  cv::namedWindow("Traffic Density", cv::WINDOW_NORMAL);

  // Main loop
  while (true) 
  {
    // Read frame from the video file
    cv::Mat img;
    bool success = cap.read(img);

    // Break the loop if the video ends or if there is an error reading the frame
    if (!success) 
    {
      break;
    }

    // Detect objects in the frame
    std::vector < int > classIds;
    std::vector < float > confidences;
    std::vector < cv::Rect > boxes;
    net.setInput(cv::dnn::blobFromImage(img, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false));
    net.forward(classIds, confidences, boxes);

    // Initialize vehicle count
    int vehicle_count = 0;

    // Process detections
    for (size_t i = 0; i < classIds.size(); i++) {
      // Check if the detected object is a vehicle (using specific class IDs for vehicles)
      // Example class IDs for different types of vehicles:
      if (classIds[i] == 2 || classIds[i] == 3 || classIds[i] == 4 || classIds[i] == 8) { // bicycle, car, motorcycle, truck
        vehicle_count++;
        // Draw a bounding box around the detected vehicle
        cv::rectangle(img, boxes[i], cv::Scalar(0, 255, 0), 2);
        // Add text for the vehicle label and confidence
        std::string label = classNames[classIds[i] - 1];
        cv::putText(img, label, cv::Point(boxes[i].x + 10, boxes[i].y + 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        std::string confidence = std::to_string(confidences[i] * 100) + "%";
        cv::putText(img, confidence, cv::Point(boxes[i].x + 200, boxes[i].y + 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      }
    }

    // Calculate traffic density (e.g., by using the number of vehicles and frame size)
    // Simple approach: Use vehicle_count to estimate traffic density
    int traffic_density = vehicle_count;

    // Display the vehicle count and traffic density on the frame
    cv::putText(img, "Vehicle Count: " + std::to_string(vehicle_count), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    cv::putText(img, "Traffic Density: " + std::to_string(traffic_density), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

    // Display the output frame
    cv::imshow("Output", img);

    // Break the loop if the 'q' key is pressed
    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  // Release the video capture and close windows
  cap.release();
  cv::destroyAllWindows();

  return 0;
}
