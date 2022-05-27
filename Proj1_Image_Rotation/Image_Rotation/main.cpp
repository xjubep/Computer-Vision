#include <iostream>
#include <opencv2\opencv.hpp>

cv::Mat problem_a_rotate_forward(cv::Mat img, double angle){
	cv::Mat output;
	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	cv::Point2d center(img.cols / 2.0, img.rows / 2.0);
	cv::Rect bbox = cv::RotatedRect(center, img.size(), angle).boundingRect();
	output = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC3);
	double rad = angle * CV_PI / 180;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{	
			// step 1 회전할 물체의 중심점을 원점으로 이동
			int x = i - center.y;
			int y = j - center.x;

			// step 2 원점을 중심으로 회전
			int rot_x = x * cos(rad) - y * sin(rad);
			int rot_y = x * sin(rad) + y * cos(rad);

			// step 3 물체를 bbox 중심으로 이동
			rot_x += bbox.height / 2.0;
			rot_y += bbox.width / 2.0;

			if (rot_x >= 0 && rot_x < bbox.height && rot_y >= 0 && rot_y < bbox.width)
				output.at<cv::Vec3b>(rot_x, rot_y) = img.at<cv::Vec3b>(i, j);
		}

	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////
	cv::imshow("a_output", output); cv::waitKey(0);
	return output;
}

cv::Mat problem_b_rotate_backward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	cv::Point2d center(img.cols / 2.0, img.rows / 2.0);
	cv::Rect bbox = cv::RotatedRect(center, img.size(), angle).boundingRect();
	output = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC3);
	double rad = angle * CV_PI / 180;

	for (int i = 0; i < bbox.height; i++)
	{
		for (int j = 0; j < bbox.width; j++)
		{
			// problem a의 step 3 inverse
			int x = i - bbox.height / 2.0;
			int y = j - bbox.width / 2.0;

			//  problem a의 step 2 inverse
			int rot_x = x * cos(rad) + y * sin(rad);
			int rot_y = -x * sin(rad) + y * cos(rad);

			//  problem a의 step 1 inverse
			rot_x += center.y;
			rot_y += center.x;

			if (rot_x >= 0 && rot_x < img.rows && rot_y >= 0 && rot_y < img.cols)
				output.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(rot_x, rot_y);
		}

	}
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("b_output", output); cv::waitKey(0);

	return output;
}

cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	cv::Point2d center(img.cols / 2.0, img.rows / 2.0);
	cv::Rect bbox = cv::RotatedRect(center, img.size(), angle).boundingRect();
	output = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC3);
	double rad = angle * CV_PI / 180;

	for (int i = 0; i < bbox.height; i++)
	{
		for (int j = 0; j < bbox.width; j++)
		{
			// problem a의 step 3 inverse
			double x = i - bbox.height / 2.0;
			double y = j - bbox.width / 2.0;

			// problem a의 step 2 inverse
			double rot_x = x * cos(rad) + y * sin(rad);
			double rot_y = -x * sin(rad) + y * cos(rad);

			// problem a의 step 1 inverse
			rot_x += center.y;
			rot_y += center.x;

			if (!(rot_x >= 0 && rot_x < img.rows && rot_y >= 0 && rot_y < img.cols)) {
				output.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				continue;
			}

			cv::Point p = cv::Point((int)rot_x, (int)rot_y);
			double p_row = rot_x - p.x;
			double p_col = rot_y - p.y;

			cv::Point q = cv::Point(p.x, p.y + 1);
			cv::Point r = cv::Point(p.x + 1, p.y);
			cv::Point s = cv::Point(p.x + 1, p.y + 1);

			if (p.x == img.rows - 1) {
				r = p;
				s = q;
			}

			if (p.y == img.cols - 1) {
				q = p;
				s = r;
			}

			output.at<cv::Vec3b>(i, j) = (1 - p_row) * (1 - p_col) * img.at<cv::Vec3b>(p.x, p.y) + (1 - p_row) * p_col * img.at<cv::Vec3b>(q.x, q.y)
				+ p_row * (1 - p_col) * img.at<cv::Vec3b>(r.x, r.y) + p_row * p_col * img.at<cv::Vec3b>(s.x, s.y);

		}

	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

cv::Mat Example_change_brightness(cv::Mat img, int num, int x, int y) {
	/*
	img : input image
	num : number for brightness (increase or decrease)
	x : x coordinate of image (for square part)
	y : y coordinate of image (for square part)

	*/
	cv::Mat output = img.clone();
	int size = 100;
	int height = (y + 100 > img.rows) ? img.rows : y + 100;
	int width = (x + 100 > img.cols) ? img.cols : x + 100;

	for (int i = x; i < width; i++)
	{
		for (int j = y; j < height; j++)
		{
			for (int c = 0; c < img.channels(); c++)
			{
				int t = img.at<cv::Vec3b>(i, j)[c] + num;
				output.at<cv::Vec3b>(i, j)[c] = t > 255 ? 255 : t < 0 ? 0 : t;
			}
		}

	}
	cv::imshow("output1", img);
	cv::imshow("output2", output);
	cv::waitKey(0);
	return output;
}

int main(void){

	double angle = 45.0f;

	cv::Mat input = cv::imread("lena.jpg");
	//Fill problem_a_rotate_forward and show output
	problem_a_rotate_forward(input, angle);
	////Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	////Fill problem_c_rotate_backward_interarea and show output
	problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	//Example_change_brightness(input, 100, 50, 125);
}