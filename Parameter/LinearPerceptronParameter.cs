namespace Classification.Parameter
{
    public class LinearPerceptronParameter : Parameter
    {
        protected double learningRate;
        protected double etaDecrease;
        protected double crossValidationRatio;
        private readonly int _epoch;

        /**
         * <summary> Parameters of the linear perceptron algorithm.</summary>
         *
         * <param name="seed">                Seed is used for random number generation.</param>
         * <param name="learningRate">        Double value for learning rate of the algorithm.</param>
         * <param name="etaDecrease">         Double value for decrease in eta of the algorithm.</param>
         * <param name="crossValidationRatio">Double value for cross validation ratio of the algorithm.</param>
         * <param name="epoch">               Integer value for epoch number of the algorithm.</param>
         */
        public LinearPerceptronParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio,
            int epoch) : base(seed)
        {
            this.learningRate = learningRate;
            this.etaDecrease = etaDecrease;
            this.crossValidationRatio = crossValidationRatio;
            this._epoch = epoch;
        }

        /**
         * <summary> Accessor for the learningRate.</summary>
         *
         * <returns>The learningRate.</returns>
         */
        public double GetLearningRate()
        {
            return learningRate;
        }

        /**
         * <summary> Accessor for the etaDecrease.</summary>
         *
         * <returns>The etaDecrease.</returns>
         */
        public double GetEtaDecrease()
        {
            return etaDecrease;
        }

        /**
         * <summary> Accessor for the crossValidationRatio.</summary>
         *
         * <returns>The crossValidationRatio.</returns>
         */
        public double GetCrossValidationRatio()
        {
            return crossValidationRatio;
        }

        /**
         * <summary> Accessor for the epoch.</summary>
         *
         * <returns>The epoch.</returns>
         */
        public int GetEpoch()
        {
            return _epoch;
        }
    }
}