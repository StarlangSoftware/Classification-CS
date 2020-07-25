namespace Classification.StatisticalTest
{
    public class StatisticalTestResult
    {
        private readonly double _pValue;
        private readonly bool _onlyTwoTailed;

        public StatisticalTestResult(double pValue, bool onlyTwoTailed)
        {
            this._pValue = pValue;
            this._onlyTwoTailed = onlyTwoTailed;
        }

        public StatisticalTestResultType OneTailed(double alpha)
        {
            if (_pValue < alpha)
            {
                return StatisticalTestResultType.REJECT;
            }

            return StatisticalTestResultType.FAILED_TO_REJECT;
        }

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

        public double GetPValue()
        {
            return _pValue;
        }
    }
}