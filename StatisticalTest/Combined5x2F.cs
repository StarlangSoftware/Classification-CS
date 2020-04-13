using Classification.Performance;
using Math;

namespace Classification.StatisticalTest
{
    public class Combined5x2F : PairedTest
    {
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
                double mean = (difference[2 * i] + difference[2 * i + 1]) / 2;
                double variance = (difference[2 * i] - mean) * (difference[2 * i] - mean) +
                                  (difference[2 * i + 1] - mean) * (difference[2 * i + 1] - mean);
                denominator += variance;
            }

            denominator *= 2;

            return numerator / denominator;
        }

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