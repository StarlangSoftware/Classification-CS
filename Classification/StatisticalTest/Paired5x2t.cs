using Classification.Performance;
using Math;

namespace Classification.StatisticalTest
{
    public class Paired5x2t : PairedTest
    {
        /// <summary>
        /// Calculates the test statistic of the 5x2 t test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Given the performances of two classifiers, the test statistic of the 5x2 t test.</returns>
        private double TestStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2)
        {
            var difference = new double[classifier1.NumberOfExperiments()];
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                difference[i] = classifier1.GetErrorRate(i) - classifier2.GetErrorRate(i);
            }

            double denominator = 0;
            for (var i = 0; i < classifier1.NumberOfExperiments() / 2; i++)
            {
                var mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
                var variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) +
                               (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
                denominator += variance;
            }

            denominator = System.Math.Sqrt(denominator / 5);

            return difference[0] / denominator;
        }

        /// <summary>
        /// Compares two classification algorithms based on their performances (accuracy or error rate) using 5x2 t test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Statistical test result of the comparison.</returns>
        public override StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2)
        {
            var statistic = TestStatistic(classifier1, classifier2);

            var degreeOfFreedom = classifier1.NumberOfExperiments() / 2;
            return new StatisticalTestResult(Distribution.TDistribution(statistic, degreeOfFreedom), false);
        }
    }
}