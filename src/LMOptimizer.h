//
// Created by hryts on 10.04.21.
//

#ifndef IMPROVEDSTITCHING_LMOPTIMIZER_H
#define IMPROVEDSTITCHING_LMOPTIMIZER_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <utility>

#include "StitcherHelper.h"

class LMOptimizer {
public:
	LMOptimizer(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& measured_values,
			 Eigen::VectorXd& parameters)
	            : m_parameters(parameters),
	              m_lm_functor(measured_values.size(), 9, measured_values)
	{}

	Eigen::Matrix3d optimize()
	{
		Eigen::LevenbergMarquardt<LMFunctor, double> lm(m_lm_functor);
//		lm.parameters.xtol = 1.0e-9;
//		lm.parameters.ftol = 1.0e-9;
//		lm.parameters.gtol = 1.0e-9;
//		lm.parameters.maxfev = 2000;
		lm.minimize(m_parameters);
 		Eigen::Matrix3d result;
		result << m_parameters(0), m_parameters(3),  m_parameters(6),
				m_parameters(1), m_parameters(4),  m_parameters(7),
				m_parameters(2), m_parameters(5),  m_parameters(8);
		return result;
	}

private:
	// Functor provides subroutines to compute error function and Jacobian for LM algorithm
	struct LMFunctor
	{
		LMFunctor(size_t number_of_values,
				  size_t number_of_parameters,
				  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> measured_values
				  )
				  : number_of_data_points(number_of_values),
				  number_of_parameters(number_of_parameters),
				  m_measured_values(std::move(measured_values))
		{}

		// Compute vector of errors
		int
		operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
		{

			Eigen::Matrix3d transformation;
			transformation << x(0), x(3),  x(6),
							  x(1), x(4),  x(7),
							  x(2), x(5),  x(8);

			std::vector<double> residuals;
			double residuals_mean = 0;
			double standard_deviation;

			for (int i = 0; i < values(); ++i) {
				Eigen::Vector3d x_value, y_value;
				y_value << m_measured_values[i].first, 1;
				x_value << m_measured_values[i].second, 1;
				double residual = (y_value - transformation * x_value).norm();
				residuals.emplace_back(residual);
				residuals_mean += residual;
			}
			residuals_mean /= (double)values();
			Eigen::Vector2d residuals_means(residuals_mean, residuals_mean);

			standard_deviation = sqrt(StitcherHelper::covariance(residuals, residuals, residuals_means));

			for (int i = 0; i < values(); ++i) {
				fvec(i) = huber(residuals[i], standard_deviation);
			}
			return 0;
		}

		// Compute the Jacobian matrix
		[[maybe_unused]]
		int
		df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
		{
			Eigen::VectorXd fvecDiff(values());

			for (int i = 0; i < x.size(); ++i)
			{
				Eigen::VectorXd xPlus(x);
				xPlus(i) += KEpsilon;

				Eigen::VectorXd xMinus(x);
				xMinus(i) -= KEpsilon;

				Eigen::VectorXd fvecPlus(values());
				operator()(xPlus, fvecPlus);

				Eigen::VectorXd fvecMinus(values());
				operator()(xMinus, fvecMinus);

				fvecDiff = (fvecPlus - fvecMinus) / (2.0f * KEpsilon);
				fjac.block(0, i, values(), 1) = fvecDiff;
			}
			return 0;
		}

		static double
		huber(double r, double standard_deviation=1, double delta_coefficient=1.345f)
		{
			double delta = delta_coefficient * standard_deviation;
			if (r < delta)
				return r * r / 2;
			return delta * r - delta * delta / 2;
		}

		[[nodiscard]] int
		values() const
		{
			return number_of_data_points;
		}

		[[maybe_unused]] [[nodiscard]] int
		inputs() const
		{
			return number_of_parameters;
		}

		const double KEpsilon = 1e-5f;
		size_t number_of_data_points;
		size_t number_of_parameters;
		std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> m_measured_values;
	};

	Eigen::VectorXd m_parameters;
	LMFunctor m_lm_functor;
};


#endif //IMPROVEDSTITCHING_LMOPTIMIZER_H
