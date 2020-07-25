using Classification.Performance;
using Math;

namespace Classification.StatisticalTest
{
    public class Combined5x2t : PairedTest
    {
        private double TestStatistic(ExperimentPerformance classifier1, ExperimentPerformance classifier2)
        {
            var difference = new double[classifier1.NumberOfExperiments()];
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                difference[i] = classifier1.GetErrorRate(i) - classifier2.GetErrorRate(i);
            }

            double denominator = 0;
            double numerator = 0;
            for (var i = 0; i < classifier1.NumberOfExperiments() / 2; i++)
            {
                var mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
                numerator += mean;
                var variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) +
                               (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
                denominator += variance;
            }

            numerator = System.Math.Sqrt(10) * numerator / 5;
            denominator = System.Math.Sqrt(denominator / 5);

            return numerator / denominator;
        }

        public override StatisticalTestResult Compare(ExperimentPerformance classifier1,
                ExperimentPerformance classifier2)
        {
            var statistic = TestStatistic(classifier1, classifier2);

            var degreeOfFreedom = classifier1.NumberOfExperiments() / 2;
            return new StatisticalTestResult(Distribution.TDistribution(statistic, degreeOfFreedom), false);
        }
    }
}