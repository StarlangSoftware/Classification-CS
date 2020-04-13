using System.Collections.Generic;

namespace Classification.Attribute
{
    public class ContinuousAttribute : Attribute
    {
        private double _value;

        /**
         * <summary>Constructor for a continuous attribute.</summary>
         *
         * <param name="value">Value of the attribute.</param>
         */
        public ContinuousAttribute(double value)
        {
            this._value = value;
        }

        /**
         * <summary>Accessor method for value.</summary>
         *
         * <returns>value</returns>
         */
        public double GetValue()
        {
            return _value;
        }

        /**
         * <summary>Mutator method for value</summary>
         *
         * <param name="value">New value of value.</param>
         */
        public void SetValue(double value)
        {
            this._value = value;
        }

        /**
         * <summary>Converts value to {@link String}.</summary>
         *
         * <returns>String representation of value.</returns>
         */
        public override string ToString()
        {
            return _value.ToString("F4");
        }

        public override int ContinuousAttributeSize()
        {
            return 1;
        }

        public override List<double> ContinuousAttributes()
        {
            var result = new List<double> {_value};
            return result;
        }
    }
}