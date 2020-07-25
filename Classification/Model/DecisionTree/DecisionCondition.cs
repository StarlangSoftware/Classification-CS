using Classification.Attribute;

namespace Classification.Model.DecisionTree
{
    public class DecisionCondition
    {
        private readonly int _attributeIndex = -1;
        private readonly char _comparison;
        private readonly Attribute.Attribute _value;

        /**
         * <summary> A constructor that sets attributeIndex and {@link Attribute} value. It also assigns equal sign to the comparison character.</summary>
         *
         * <param name="attributeIndex">Integer number that shows attribute index.</param>
         * <param name="value">         The value of the {@link Attribute}.</param>
         */
        public DecisionCondition(int attributeIndex, Attribute.Attribute value)
        {
            this._attributeIndex = attributeIndex;
            _comparison = '=';
            this._value = value;
        }

        /**
         * <summary> A constructor that sets attributeIndex, comparison and {@link Attribute} value.</summary>
         *
         * <param name="attributeIndex">Integer number that shows attribute index.</param>
         * <param name="value">         The value of the {@link Attribute}.</param>
         * <param name="comparison">    Comparison character.</param>
         */
        public DecisionCondition(int attributeIndex, char comparison, Attribute.Attribute value)
        {
            this._attributeIndex = attributeIndex;
            this._comparison = comparison;
            this._value = value;
        }

        /**
         * <summary> The satisfy method takes an {@link Instance} as an input.
         * <p/>
         * If defined {@link Attribute} value is a {@link DiscreteIndexedAttribute} it compares the index of {@link Attribute} of instance at the
         * attributeIndex and the index of {@link Attribute} value and returns the result.
         * <p/>
         * If defined {@link Attribute} value is a {@link DiscreteAttribute} it compares the value of {@link Attribute} of instance at the
         * attributeIndex and the value of {@link Attribute} value and returns the result.
         * <p/>
         * If defined {@link Attribute} value is a {@link ContinuousAttribute} it compares the value of {@link Attribute} of instance at the
         * attributeIndex and the value of {@link Attribute} value and returns the result according to the comparison character whether it is
         * less than or greater than signs.</summary>
         *
         * <param name="instance">Instance to compare.</param>
         * <returns>True if gicen instance satisfies the conditions.</returns>
         */
        public bool Satisfy(Instance.Instance instance)
        {
            if (_value is DiscreteIndexedAttribute discreteIndexedAttribute)
            {
                if (discreteIndexedAttribute.GetIndex() != -1)
                {
                    return ((DiscreteIndexedAttribute) instance.GetAttribute(_attributeIndex)).GetIndex() ==
                           discreteIndexedAttribute.GetIndex();
                }

                return true;
            }

            if (_value is DiscreteAttribute discreteAttribute) {
                return ((DiscreteAttribute) instance.GetAttribute(_attributeIndex)).GetValue()
                       == discreteAttribute.GetValue();
            }

            if (_value is ContinuousAttribute continuousAttribute)
            {
                if (_comparison == '<')
                {
                    return ((ContinuousAttribute) instance.GetAttribute(_attributeIndex)).GetValue() <=
                           continuousAttribute.GetValue();
                }

                if (_comparison == '>')
                {
                    return ((ContinuousAttribute) instance.GetAttribute(_attributeIndex)).GetValue() >
                           continuousAttribute.GetValue();
                }
            }
            return false;
        }
    }
}