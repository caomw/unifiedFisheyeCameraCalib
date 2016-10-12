#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class Problem_FisheyeUnifiedCalib {
public:
	~Problem_FisheyeUnifiedCalib() {
		delete[] observations_;
		delete[] cameraParameters_;
	}
	int num_observations()       const { return num_observations_; }
	int num_cameraParameters() const { return num_parameters_; }
	const double* observations() const { return observations_; }
	double* parameters() { return cameraParameters_; }
	
	void showFinalResults()
	{
		for (int i = 0; i < num_parameters_; i++)
		{
			std::cout << cameraParameters_[i] << " " ;
		}
		std::cout << std::endl;
	}

	bool LoadFile(const char* filename) {
		std::ifstream mDataFile;
		mDataFile.open(filename, std::ifstream::in);
		if (!mDataFile.is_open())
		{
			std::cerr << " cannot open data file for optimization... " << std::endl;
			return false;
		}
		
		mDataFile >> num_parameters_; 

		cameraParameters_ = new double[num_parameters_];

		for (int i = 0; i < num_parameters_; i++)
		{
			mDataFile >> cameraParameters_[i];
		}

		mDataFile >> num_observations_;
		observations_ = new double[num_observations_ * 5];

		for (int i = 0; i < num_observations_; i++)
		{
			mDataFile >> observations_[i * 5];
			mDataFile >> observations_[i * 5 + 1];
			mDataFile >> observations_[i * 5 + 2];
			mDataFile >> observations_[i * 5 + 3];
			mDataFile >> observations_[i * 5 + 4];
		}
		return true;
	}
private:
	int num_observations_;
	int num_parameters_;
	double* observations_; // u v xs ys zs
	double* cameraParameters_; // fx, fy, cx, cy, xi
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct cost_reprojectionError {
	cost_reprojectionError(double u, double v, double Xs, double Ys, double Zs)
		: 
		u(u), 
		v(v),
		Xs(Xs),
		Ys(Ys),
		Zs(Zs){}
	
	template <typename T>
	bool operator()(const T* const camera,
		T* residuals) const {
		
		// camera contains intrinsic parameters
		T fx = camera[0];
		T fy = camera[1];
		T xi = camera[2];
		T cx = camera[3];
		T cy = camera[4];
		T k1 = camera[5];
		T k2 = camera[6];
		T k3 = camera[7];
		T p1 = camera[8];
		T p2 = camera[9];

		// project to normalized image plane
		T mu = T(Xs) / (T(Zs) + xi);
		T mv = T(Ys) / (T(Zs) + xi);

		// add radial distortion
		T rho = sqrt(mu*mu + mv*mv);
		T rho2 = rho*rho;
		T rho4 = rho2*rho2;
		T distortion = T(1.0) + k1*rho2 + k2*rho4 + k3*rho2*rho4;

		T mdu = mu*distortion;
		T mdv = mv*distortion;

		// add tangential distortion
		mdu = mdu + T(2.0) * p1*mdu*mdv + p2*(rho2 + T(2.0) * mdu*mdu);
		mdv = mdv + p1*(rho2 + T(2.0) * mdv*mdv) + T(2.0) * p2*mdu*mdv;

		// transform to image frame
		T umodel = fx*mdu + cx;
		T vmodel = fy*mdv + cy;
		
		// The error is the difference between the predicted and observed position.
		residuals[0] = umodel - T(u);
		residuals[1] = vmodel - T(v);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double Xs, const double Ys, const double Zs) {
		return (new ceres::AutoDiffCostFunction<cost_reprojectionError, 2, 10>(
			new cost_reprojectionError(observed_x, observed_y, Xs, Ys, Zs)));
	}
	
	double u, v; // observed pixel position
	double Xs, Ys, Zs; // points on unit sphere
};

int main(int argc, char** argv) {	
	google::InitGoogleLogging(argv[0]);
	if (argc != 2) {
		std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
		return 1;
	}
	Problem_FisheyeUnifiedCalib problem_fisheyeCalib;
	if (!problem_fisheyeCalib.LoadFile(argv[1])) {
		std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
		return 1;
	}

	// Create residuals for each observation in the bundle adjustment problem. The
	// parameters for cameras and points are added automatically.
	ceres::Problem problem;

	const double* observations = problem_fisheyeCalib.observations();
	for (int i = 0; i < problem_fisheyeCalib.num_observations(); ++i) {
		// Each Residual block takes a point and a camera as input and outputs a 2
		// dimensional residual. Internally, the cost function stores the observed
		// image location and compares the reprojection against the observation.
		ceres::CostFunction* cost_function = cost_reprojectionError::Create(observations[5*i + 0], observations[5*i + 1], observations[5*i + 2], observations[5*i + 3], observations[5*i + 4]);
		problem.AddResidualBlock(cost_function,
			NULL /* squared loss */,
			problem_fisheyeCalib.parameters());
	}

	// Make Ceres automatically detect the bundle structure. Note that the
	// standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
	// for standard bundle adjustment problems.
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	problem_fisheyeCalib.showFinalResults();

	return 0;
}