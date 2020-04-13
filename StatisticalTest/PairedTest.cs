using Classification.Performance;

namespace Classification.StatisticalTest
{
    public abstract class PairedTest
    {
        public abstract StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2);

        public int Compare(ExperimentPerformance classifier1, ExperimentPerformance classifier2, double alpha)
        {
            var testResult1 = Compare(classifier1, classifier2);
            var testResult2 = Compare(classifier2, classifier1);
            var testResultType1 = testResult1.OneTailed(alpha);

            var testResultType2 = testResult2.OneTailed(alpha);
            if (testResultType1.Equals(StatisticalTestResultType.REJECT))
            {
                return 1;
            }

            if (testResultType2.Equals(StatisticalTestResultType.REJECT))
            {
                return -1;
            }

            return 0;
        }
    }
}