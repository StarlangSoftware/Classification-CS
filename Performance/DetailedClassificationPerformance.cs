namespace Classification.Performance
{
    public class DetailedClassificationPerformance : ClassificationPerformance
    {
        private readonly ConfusionMatrix _confusionMatrix;

        /**
         * <summary> A constructor that  sets the accuracy and errorRate as 1 - accuracy via given {@link ConfusionMatrix} and also sets the confusionMatrix.</summary>
         *
         * <param name="confusionMatrix">{@link ConfusionMatrix} input.</param>
         */
        public DetailedClassificationPerformance(ConfusionMatrix confusionMatrix) : base(confusionMatrix.GetAccuracy())
        {
            this._confusionMatrix = confusionMatrix;
        }

        /**
         * <summary> Accessor for the confusionMatrix.</summary>
         *
         * <returns>ConfusionMatrix.</returns>
         */
        public ConfusionMatrix GetConfusionMatrix()
        {
            return _confusionMatrix;
        }
    }
}