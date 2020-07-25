namespace Classification.Model
{
    public abstract class Model
    {
        /**
         * <summary> An abstract predict method that takes an {@link Instance} as an input.</summary>
         *
         * <param name="instance">{@link Instance} to make prediction.</param>
         * <returns>The class label as a String.</returns>
         */
        public abstract string Predict(Instance.Instance instance);
        
    }
}