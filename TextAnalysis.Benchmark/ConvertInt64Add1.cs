namespace TextAnalysis.Benchmark;

using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Neco.Common.Extensions;
using SentencePieceTokenizer;

[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class ConvertInt64Add1 {
	private Int32[] _source = null!;
	private Int64[] _target = null!;
	private Int32[] _target32 = null!;

	[Params(512, 4092)] public Int32 N;

	[GlobalSetup]
	public void Setup() {
		_source = new Int32[N];
		_target = new Int64[N + 1];
		_target32 = new Int32[N];
		for (Int32 i = 0; i < _source.Length; i++) {
			_source[i] = Random.Shared.Next(0, Int32.MaxValue);
		}
	}

	[Benchmark]
	public void Linq() {
		_source.Select(i => (Int64)i + 1).CopyTo(_target, 0);
	}

	[Benchmark]
	public void For() {
		for (Int32 i = 0; i < _source.Length; i++) {
			_target[i] = _source[i] + 1;
		}
	}

	[Benchmark]
	public void ForLocalCopy() {
		Int32[] source = _source;
		Int64[] target = _target;
		for (Int32 i = 0; i < source.Length; i++) {
			target[i] = source[i] + 1;
		}
	}

	[Benchmark]
	public void CurrentImplementation() => HugginFaceHack.ConvertToInt64AndAdd1(_source, _target);

	[Benchmark]
	public void AvxArrayInsteadOfSpan() => ConvertToInt64AndAdd1(_source, _target, 0);

	public static unsafe void ConvertToInt64AndAdd1(Int32[] tokens, Int64[] target, Int32 targetOffset) {
		ArgumentOutOfRangeException.ThrowIfLessThan(target.Length, tokens.Length + targetOffset);

		fixed (Int32* tokenPtr = tokens) {
			Int32* ptr = tokenPtr;
			Int32 tokensRemaining = tokens.Length;

			if (Avx2.IsSupported && Vector256.IsHardwareAccelerated) {
				Vector256<Int32> addMe = Vector256<Int32>.One;

				while (tokensRemaining >= Vector256<Int32>.Count) {
					Vector256<Int32> sourceTokens = Vector256.Load(ptr);
					Vector256<Int32> addResult = Avx2.Add(sourceTokens, addMe);
					(Vector256<Int64> lower, Vector256<Int64> upper) = Vector256.Widen(addResult);
					lower.StoreUnsafe(ref target[targetOffset]);
					upper.StoreUnsafe(ref target[targetOffset + Vector256<Int64>.Count]);
					targetOffset += Vector256<Int64>.Count * 2;

					ptr += Vector256<Int32>.Count;
					tokensRemaining -= Vector256<Int32>.Count;
				}
			}

			while (tokensRemaining > 0) {
				target[targetOffset++] = (Int64)((*ptr) + 1);
				++ptr;
				--tokensRemaining;
			}
		}
	}

	[Benchmark]
	public void TensorPrimitivesChecked() {
		TensorPrimitives.Add(_source, 1, _target32);
		// Checked, Truncating, Saturating makes no difference
		TensorPrimitives.ConvertChecked<Int32, Int64>(_target32, _target);
	}
}