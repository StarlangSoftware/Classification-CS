using Classification.Performance;

namespace Classification.StatisticalTest
{
    public abstract class PairedTest
    {
        public abstract StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2);

        /// <summary>
        /// Compares two classification algorithms based on their performances (accuracy or error rate). The method first
        /// checks the null hypothesis mu1 less than mu2, if the test rejects this null hypothesis with alpha level of confidence, it
        /// decides mu1 > mu2. The algorithm then checks the null hypothesis mu1 > mu2, if the test rejects that null
        /// hypothesis with alpha level of confidence, if decides mu1 less than mu2. If none of the two tests are rejected, it can not
        /// make a decision about the performances of algorithms.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <param name="alpha">Alpha level defined for the statistical test.</param>
        /// <returns>1 if the performance of the first algorithm is larger than the second algorithm, -1 if the performance of
        /// the second algorithm is larger than the first algorithm, 0 if they have similar performance.</returns>
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