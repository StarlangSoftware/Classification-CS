using Classification.Performance;
using Math;

namespace Classification.StatisticalTest
{
    public class Combined5x2F : PairedTest
    {
        /// <summary>
        /// Calculates the test statistic of the combined 5x2 cv F test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Given the performances of two classifiers, the test statistic of the combined 5x2 cv F test.</returns>
        private double TestStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2)
        {
            var difference = new double[classifier1.NumberOfExperiments()];

            double numerator = 0;
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                difference[i] = classifier1.GetErrorRate(i) - classifier2.GetErrorRate(i);
                numerator += difference[i] * difference[i];
            }

            double denominator = 0;
            for (var i = 0; i < classifier1.NumberOfExperiments() / 2; i++)
            {
                var mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
                var variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) +
                               (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
                denominator += variance;
            }

            denominator *= 2;

            return numerator / denominator;
        }

        /// <summary>
        /// Compares two classification algorithms based on their performances (accuracy or error rate) using combined 5x2 cv F test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Statistical test result of the comparison.</returns>
        public override StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2)
        {
            var statistic = TestStatistic(classifier1, classifier2);
            var degreeOfFreedom1 = classifier1.NumberOfExperiments();

            var degreeOfFreedom2 = classifier1.NumberOfExperiments() / 2;
            return new StatisticalTestResult(Distribution.FDistribution(statistic, degreeOfFreedom1, degreeOfFreedom2),
                true
            );
        }
    }
}