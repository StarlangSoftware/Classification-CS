using System.Collections.Generic;

namespace Classification.Parameter
{
    public class DeepNetworkParameter : LinearPerceptronParameter
    {
        private readonly List<int> _hiddenLayers;
        private readonly ActivationFunction _activationFunction;

        /**
         * <summary> Parameters of the deep network classifier.</summary>
         *
         * <param name="seed">                Seed is used for random number generation.</param>
         * <param name="learningRate">        Double value for learning rate of the algorithm.</param>
         * <param name="etaDecrease">         Double value for decrease in eta of the algorithm.</param>
         * <param name="crossValidationRatio">Double value for cross validation ratio of the algorithm.</param>
         * <param name="epoch">               Integer value for epoch number of the algorithm.</param>
         * <param name="hiddenLayers">        An integer {@link ArrayList} for hidden layers of the algorithm.</param>
         * <param name="activationFunction">         Activation function.</param>
         */
        public DeepNetworkParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio,
            int epoch, List<int> hiddenLayers, ActivationFunction activationFunction) : base(seed, learningRate, etaDecrease, crossValidationRatio, epoch)
        {
            this._hiddenLayers = hiddenLayers;
            this._activationFunction = activationFunction;
        }

        /**
         * <summary> The layerSize method returns the size of the hiddenLayers {@link ArrayList}.</summary>
         *
         * <returns>The size of the hiddenLayers {@link ArrayList}.</returns>
         */
        public int LayerSize()
        {
            return _hiddenLayers.Count;
        }

        /**
         * <summary> The getHiddenNodes method takes a layer index as an input and returns the element at the given index of hiddenLayers
         * {@link ArrayList}.</summary>
         *
         * <param name="layerIndex">Index of the layer.</param>
         * <returns>The element at the layerIndex of hiddenLayers {@link ArrayList}.</returns>
         */
        public int GetHiddenNodes(int layerIndex)
        {
            return _hiddenLayers[layerIndex];
        }
        
        /**
         * <summary> Accessor for the activation function.</summary>
         *
         * <returns>The activation function.</returns>
         */
        public ActivationFunction GetActivationFunction()
        {
            return _activationFunction;
        }
        
    }
}