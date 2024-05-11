namespace Classification.StatisticalTest
{
    public class StatisticalTestResult
    {
        private readonly double _pValue;
        private readonly bool _onlyTwoTailed;

        /// <summary>
        /// Constructor of the StatisticalTestResult. It sets the attribute values.
        /// </summary>
        /// <param name="pValue">p value of the statistical test result</param>
        /// <param name="onlyTwoTailed">True, if this test applicable only two tailed tests, false otherwise.</param>
        public StatisticalTestResult(double pValue, bool onlyTwoTailed)
        {
            this._pValue = pValue;
            this._onlyTwoTailed = onlyTwoTailed;
        }

        /// <summary>
        /// Returns reject or failed to reject, depending on the alpha level and p value of the statistical test that checks
        /// one tailed null hypothesis such as mu1 less than mu2. If p value is less than the alpha level, the test rejects the null
        /// hypothesis. Otherwise, it fails to reject the null hypothesis.
        /// </summary>
        /// <param name="alpha">Alpha level of the test</param>
        /// <returns>If p value is less than the alpha level, the test rejects the null hypothesis. Otherwise, it fails to
        /// reject the null hypothesis.</returns>
        public StatisticalTestResultType OneTailed(double alpha)
        {
            if (_pValue < alpha)
            {
                return StatisticalTestResultType.REJECT;
            }

            return StatisticalTestResultType.FAILED_TO_REJECT;
        }

        /// <summary>
        /// Returns reject or failed to reject, depending on the alpha level and p value of the statistical test that checks
        /// one tailed null hypothesis such as mu1 less than mu2 or two tailed null hypothesis such as mu1 = mu2. If the null
        /// hypothesis is two tailed, and p value is less than the alpha level, the test rejects the null hypothesis.
        /// Otherwise, it fails to reject the null hypothesis. If the null  hypothesis is one tailed, and p value is less
        /// than alpha / 2 or p value is larger than 1 - alpha / 2, the test  rejects the null  hypothesis. Otherwise, it
        /// fails to reject the null hypothesis.
        /// </summary>
        /// <param name="alpha">Alpha level of the test</param>
        /// <returns>If the null  hypothesis is two tailed, and p value is less than the alpha level, the test rejects the
        /// null hypothesis.  Otherwise, it fails to reject the null hypothesis. If the null  hypothesis is one tailed, and
        /// p value is less  than alpha / 2 or p value is larger than 1 - alpha / 2, the test  rejects the null  hypothesis.
        /// Otherwise, it  fails to reject the null hypothesis.</returns>
        public StatisticalTestResultType TwoTailed(double alpha)
        {
            if (_onlyTwoTailed)
            {
                if (_pValue < alpha)
                {
                    return StatisticalTestResultType.REJECT;
                }

                return StatisticalTestResultType.FAILED_TO_REJECT;
            }

            if (_pValue < alpha / 2 || _pValue > 1 - alpha / 2)
            {
                return StatisticalTestResultType.REJECT;
            }

            return StatisticalTestResultType.FAILED_TO_REJECT;
        }

        /// <summary>
        /// Accessor for the p value of the statistical test result.
        /// </summary>
        /// <returns>p value of the statistical test result</returns>
        public double GetPValue()
        {
            return _pValue;
        }
    }
}