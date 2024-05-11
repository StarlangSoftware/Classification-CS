using Classification.Performance;
using Math;

namespace Classification.StatisticalTest
{
    public class Pairedt : PairedTest
    {
        /// <summary>
        /// Calculates the test statistic of the paired t test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Given the performances of two classifiers, the test statistic of the paired t test.</returns>
        private double TestStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2)
        {
            var difference = new double[classifier1.NumberOfExperiments()];
            var sum = 0.0;
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                difference[i] = classifier1.GetErrorRate(i) - classifier2.GetErrorRate(i);
                sum += difference[i];
            }

            var mean = sum / classifier1.NumberOfExperiments();
            sum = 0.0;
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                sum += (difference[i] - mean) * (difference[i] - mean);
            }

            var standardDeviation = System.Math.Sqrt(sum / (classifier1.NumberOfExperiments() - 1));
            return System.Math.Sqrt(classifier1.NumberOfExperiments()) * mean / standardDeviation;
        }

        /// <summary>
        /// Compares two classification algorithms based on their performances (accuracy or error rate) using paired t test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Statistical test result of the comparison.</returns>
        public override StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2)
        {
            var statistic = TestStatistic(classifier1, classifier2);

            var degreeOfFreedom = classifier1.NumberOfExperiments() - 1;
            return new StatisticalTestResult(Distribution.TDistribution(statistic, degreeOfFreedom), false);
        }
    }
}