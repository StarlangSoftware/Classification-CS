using System.Collections.Generic;

namespace Classification.Attribute
{
    public class DiscreteAttribute : Attribute
    {
        private readonly string _value;

        /**
         * <summary>Constructor for a discrete attribute.</summary>
         *
         * <param name="value">Value of the attribute.</param>
         */
        public DiscreteAttribute(string value)
        {
            this._value = value;
        }

        /**
         * <summary>Accessor method for value.</summary>
         *
         * <returns>value</returns>
         */
        public string GetValue()
        {
            return _value;
        }

        /**
         * <summary>Converts value to {@link string}.</summary>
         *
         * <returns>string representation of value.</returns>
         */
        public override string ToString()
        {
            if (_value == ",")
            {
                return "comma";
            }

            return _value;
        }


        public override int ContinuousAttributeSize()
        {
            return 0;
        }

        public override List<double> ContinuousAttributes()
        {
            return new List<double>();
        }
    }
}