//
// Created by hryts on 12.04.21.
//

#ifndef IMPROVEDSTITCHING_MYLMOPTIMIZER_H
#define IMPROVEDSTITCHING_MYLMOPTIMIZER_H


class MyLMOptimizer {
public:
	LMOptimizer(const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& measured_values,
	            Eigen::VectorXf& parameters)
			: m_parameters(parameters), m_measured_values(measured_values)
	{}
	static int computeErrors(const Eigen::VectorXf &parameters, Eigen::VectorXf &residuals)
	{
		Eigen::Matrix3f transformation;
		transformation << parameters(0), parameters(3),  parameters(6),
						  parameters(1), parameters(4),  parameters(7),
						  parameters(2), parameters(5),  parameters(8);
//			std::cout << "transformation: " << std::endl << transformation << std::endl;
		std::vector<float> residuals;
		float residuals_mean = 0;
		float standard_deviation = 0;

		for (int i = 0; i < values(); ++i) {
			Eigen::Vector3f x_value, y_value;
			y_value << m_measured_values[i].first, 1;
			x_value << m_measured_values[i].second, 1;
			float residual = (y_value - transformation * x_value).norm();
			residuals.emplace_back(residual);
			residuals_mean += residual;
		}
		residuals_mean /= values();
		Eigen::Vector2f residuals_means(residuals_mean, residuals_mean);

		standard_deviation = sqrt(covariance(residuals, residuals, residuals_means));

		for (int i = 0; i < values(); ++i) {
			residuals(i) = huber(residuals[i], standard_deviation);
		}
		return 0;
	}

	static int computeJacobian(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac)
	{
		Eigen::VectorXf fvecDiff(values());

		for (int i = 0; i < x.size(); ++i) {
			Eigen::VectorXf xPlus(x);
			xPlus(i) += KEpsilon;
			Eigen::VectorXf xMinus(x);
			xMinus(i) -= KEpsilon;
			Eigen::VectorXf fvecPlus(values());
			computeErrors(xPlus, fvecPlus);
			Eigen::VectorXf fvecMinus(values());
			computeErrors(xMinus, fvecMinus);
			fvecDiff = (fvecPlus - fvecMinus) / (2.0f * KEpsilon);
			fjac.block(0, i, values(), 1) = fvecDiff;
		}
		return 0;
	}

	void optimize()
	{

	}

private:
	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> m_measured_values;
};


#endif //IMPROVEDSTITCHING_MYLMOPTIMIZER_H
