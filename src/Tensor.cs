using System;
using MiniECS;

namespace Backend.EvoLearning {

    [Serializable]
    public sealed class TensorDTO {
        public int ShapeX;
        public int ShapeY;
        public float[] Buffer;

        public bool IsSame(TensorDTO rhs) {
            if (ShapeX != rhs.ShapeX) return false;
            if (ShapeY != rhs.ShapeY) return false;
            for (var i = 0; i < Buffer.Length; i++) {
                if (Buffer[i] != rhs.Buffer[i]) {
                    return false;
                }
            }
            return true;
        }
    }

    public sealed class Tensor {

        private readonly int _shapeX;
        private readonly int _shapeY;

        private readonly float[] _buffer;

        public int ShapeX { get { return _shapeX; } }
        public int ShapeY { get { return _shapeY; } }
        public string Shape { get { return "(" + _shapeX + ", " + _shapeY + ")"; } }
        public int TotalCells {  get { return _shapeX * _shapeY; } }

        public Tensor(int shapeX, int shapeY, float[] buffer = null) {
            _shapeX = shapeX;
            _shapeY = shapeY;
            var size = shapeX * shapeY;
            if (buffer != null) {
                Pre.Assert(buffer.Length == size, buffer.Length, size);
                _buffer = buffer;
            } else {
                _buffer = new float[size];
            }
        }

        public static Tensor NewFromDTO(TensorDTO dto) {
            var result = new Tensor(dto.ShapeX, dto.ShapeY);
            result.CopyFrom(dto.Buffer);
            return result;
        }

        public static Tensor NewFromRandom(int shapeX, int shapeY, Random random = null) {
            var result = new Tensor(shapeX, shapeY);
            if (random == null) {
                random = new Random();
            }
            var length = result._buffer.Length;
            for (var i = 0; i < length; i++) {
                result._buffer[i] = (float) random.NextDouble();
            }
            return result;
        }

        public TensorDTO ToDTO() {
            return new TensorDTO() {
                ShapeX = _shapeX,
                ShapeY = _shapeY,
                Buffer = _buffer
            };
        }

        public void CopyFrom(float[] values) {
            Pre.Assert(values.Length == _buffer.Length, values.Length, _buffer.Length);
            var length = _buffer.Length;
            for (var i = 0; i < length; i++) {
                _buffer[i] = values[i];
            }
        }

        public void Set(int x, int y, float value)
        {
            Pre.Assert(x >= 0 && x < _shapeX, x, _shapeX);
            Pre.Assert(y >= 0 && y < _shapeY, y, _shapeY);
            _buffer[x + y * _shapeX] = value;
        }

        public float Get(int x, int y)
        {
            Pre.Assert(x >= 0 && x < _shapeX, x, _shapeX);
            Pre.Assert(y >= 0 && y < _shapeY, y, _shapeY);
            return _buffer[x + y * _shapeX];
        }

        public Tensor Reshape(int shapeX, int shapeY) {
            var result = new Tensor(shapeX, shapeY);
            result.CopyFrom(_buffer);
            return result;
        }

        public Tensor T()
        {
            var t = new Tensor(_shapeY, _shapeX);
            t.CopyFrom(_buffer);
            return t;
        }

        public float SumAll()
        {
            var result = 0.0;
            var length = _buffer.Length;
            for (var i = 0; i < length; i++)
            {
                result += _buffer[i];
            }
            return (float) result;
        }

        public Tensor Apply(Func<float, float> func) {
            var result = new Tensor(_shapeX, _shapeY);
            var length = _buffer.Length;
            for (var i = 0; i < length; i++) {
                result._buffer[i] = func(_buffer[i]);
            }
            return result;
        }

        public static Tensor operator +(Tensor lhs, Tensor rhs)
        {
            return SameDimensionsOp(lhs, rhs, (a, b) => a + b);
        }

        public static Tensor operator -(Tensor lhs, Tensor rhs)
        {
            return SameDimensionsOp(lhs, rhs, (a, b) => a - b);
        }

        public Tensor Compare(Tensor rhs)
        {
            return SameDimensionsOp(this, rhs, (a, b) => Math.Max(a, b) - Math.Min(a, b));
        }

        private static Tensor SameDimensionsOp(Tensor lhs, Tensor rhs, Func<float, float, float> func) {
            Pre.Assert(lhs._shapeX == rhs._shapeX, lhs._shapeX, rhs._shapeX);
            Pre.Assert(lhs._shapeY == rhs._shapeY, lhs._shapeY, rhs._shapeY);
            var result = new Tensor(lhs._shapeX, lhs._shapeY);
            var length = result._buffer.Length;
            for (var i = 0; i < length; i++)
            {
                result._buffer[i] = func(lhs._buffer[i], rhs._buffer[i]);
            }
            return result;
        }

        public Tensor BroadcastSum(Tensor rhs) {
            return BroadcastOp(rhs, (a, b) => a + b);
        }

        public Tensor BroadcastSubstract(Tensor rhs) {
            return BroadcastOp(rhs, (a, b) => a - b);
        }

        public Tensor BroadcastMult(Tensor rhs) {
            return BroadcastOp(rhs, (a, b) => a * b);
        }

        public Tensor BroadcastDivide(Tensor rhs) {
            return BroadcastOp(rhs, (a, b) => a / b);
        }

        private Tensor BroadcastOp(Tensor rhs, Func<float, float, float> func) {
            var extendRows = _shapeX == rhs._shapeX;
            var extendColumns = _shapeY == rhs._shapeY;
            Pre.Assert(extendRows || extendColumns, extendRows, extendColumns);
            Pre.Assert(extendRows == false && extendColumns || extendColumns == false && extendRows, extendRows, extendColumns);
            var result = new Tensor(_shapeX, _shapeY);
            result.CopyFrom(_buffer);
            if (extendRows) {
                Pre.Assert(rhs._shapeY == 1, rhs._shapeY);
                for (var x = 0; x < _shapeX; x++)
                {
                    for (var y = 0; y < _shapeY; y++)
                    {
                        result._buffer[x + y * _shapeX] = func(result._buffer[x + y * _shapeX], rhs._buffer[x]);
                    }
                }
            } else {
                Pre.Assert(rhs._shapeX == 1, rhs._shapeX);
                for (var x = 0; x < _shapeX; x++)
                {
                    for (var y = 0; y < _shapeY; y++)
                    {
                        result._buffer[x + y * _shapeX] = func(result._buffer[x + y * _shapeX], rhs._buffer[y]);
                    }
                }
            }
            return result;
        }

        public Tensor Dot(Tensor rhs) {
            Pre.Assert(_shapeY == rhs._shapeX, _shapeY, rhs._shapeX);
            var n = _shapeX;
            var m = _shapeY;
            var p = rhs._shapeY;
            var result = new Tensor(n, p);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (var k = 0; k < m; k++)
                    {
                        sum += _buffer[i + k * n] * rhs._buffer[k + j * m];
                    }
                    result._buffer[i + j * p] = sum;
                }
            }
            return result;
        }

        private static Tensor ScalarOp(Tensor lhs, float n, Func<float, float, float> func) {
            var result = new Tensor(lhs._shapeX, lhs._shapeY);
            var length = result._buffer.Length;
            for (var i = 0; i < length; i++)
            {
                result._buffer[i] = func(lhs._buffer[i], n);
            }
            return result;            
        }

        public static Tensor operator +(Tensor lhs, float n)
        {
            return ScalarOp(lhs, n, (a, b) => a + b);
        }

        public static Tensor operator -(Tensor lhs, float n)
        {
            return ScalarOp(lhs, n, (a, b) => a - b);
        }

        public static Tensor operator *(Tensor lhs, float n)
        {
            return ScalarOp(lhs, n, (a, b) => a * b);
        }

        public static Tensor operator /(Tensor lhs, float n)
        {
            return ScalarOp(lhs, n, (a, b) => a / b);
        }

        public Tensor Clone()
        {
            var t = new Tensor(_shapeX, _shapeY);
            t.CopyFrom(_buffer);
            return t;
        }

        public override string ToString() {
            var result = "[";
            var length = _buffer.Length;
            for (var x = 0; x < _shapeX; x++)
            {
                if (x != 0)
                {
                    result += ", ";
                }
                result += "(";
                for (var y = 0; y < _shapeY; y++)
                {
                    if (y != 0) {
                        result += ", ";
                    }
                    result += _buffer[x + y * _shapeX];
                }
                result += ")";
            }
            result += "]";
            return result;
        }
    }
}