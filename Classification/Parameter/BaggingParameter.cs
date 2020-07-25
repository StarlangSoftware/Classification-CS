namespace Classification.Parameter
{
    public class BaggingParameter : Parameter
    {
        protected int ensembleSize;

        /**
         * <summary>Parameters of the bagging trees algorithm.</summary>
         *
         * <param name="seed">        Seed is used for random number generation.</param>
         * <param name="ensembleSize">The number of trees in the bagged forest.</param>
         */
        public BaggingParameter(int seed, int ensembleSize) : base(seed)
        {
            this.ensembleSize = ensembleSize;
        }

        /**
         * <summary>Accessor for the ensemble size.</summary>
         *
         * <returns>The ensemble size.</returns>
         */
        public int GetEnsembleSize()
        {
            return ensembleSize;
        }
    }
}