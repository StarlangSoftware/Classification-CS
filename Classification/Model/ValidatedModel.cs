using Classification.Performance;

namespace Classification.Model
{
    public abstract class ValidatedModel : Model
    {
        /**
         * <summary> The testClassifier method takes an {@link InstanceList} as an input and returns an accuracy value as {@link ClassificationPerformance}.</summary>
         *
         * <param name="data">{@link InstanceList} to test.</param>
         * <returns>Accuracy value as {@link ClassificationPerformance}.</returns>
         */
        public ClassificationPerformance TestClassifier(InstanceList.InstanceList data)
        {
            double total = data.Size();
            var count = 0;
            for (var i = 0; i < data.Size(); i++)
            {
                if (data.Get(i).GetClassLabel() == Predict(data.Get(i)))
                {
                    count++;
                }
            }

            return new ClassificationPerformance(count / total);
        }
    }
}