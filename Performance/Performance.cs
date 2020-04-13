namespace Classification.Performance
{
    public class Performance
    {
        protected double errorRate;

        /**
         * <summary> Constructor that sets the error rate.</summary>
         *
         * <param name="errorRate">Double input.</param>
         */
        public Performance(double errorRate)
        {
            this.errorRate = errorRate;
        }

        /**
         * <summary> Accessor for the error rate.</summary>
         *
         * <returns>Double errorRate.</returns>
         */
        public double GetErrorRate()
        {
            return errorRate;
        }
    }
}