namespace TextAnalysis;

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Neco.Common.Extensions;

public sealed class InitializerReference : IDisposable {
	public readonly Int32 DataOffset;
	public readonly Int32 DataLength;
	public readonly List<CachedOrtValue> CachedOrtValues = new(1);

	public InitializerReference(Int32 dataOffset, Int32 dataLength) {
		DataOffset = dataOffset;
		DataLength = dataLength;
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		foreach (CachedOrtValue cachedOrtValue in CachedOrtValues) {
			cachedOrtValue.Dispose();
		}

		CachedOrtValues.Clear();
	}

	#endregion
}

public readonly struct CachedOrtValue : IDisposable {
	public readonly OrtValue Value;
	public readonly TensorElementType Type;
	public readonly Int64[]? Shape;

	public CachedOrtValue(OrtValue value, TensorElementType type, Int64[]? shape) {
		Value = value;
		Type = type;
		Shape = shape;
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		Value.Dispose();
	}

	#endregion
}

// TODO: Dispose and cleanup not properly implemented yet
public sealed class OrtValueCache : IDisposable {
	private readonly Dictionary<String, InitializerReference> _cache = new();
	private byte[] _initializerBlob = [];
	private Int32 _currentDataOffset = 0;
	private string _path;

	private Int64[] InitializerBlobInt64 => Unsafe.As<Byte[], Int64[]>(ref _initializerBlob);
	private Int32[] InitializerBlobInt32 => Unsafe.As<Byte[], Int32[]>(ref _initializerBlob);
	private Double[] InitializerBlobDouble => Unsafe.As<Byte[], Double[]>(ref _initializerBlob);
	private Single[] InitializerBlobSingle => Unsafe.As<Byte[], Single[]>(ref _initializerBlob);

	private Boolean TryGetCachedValue(String file, TensorElementType type, Int64[]? shape, [NotNullWhen(true)] out OrtValue? cachedOrtValue) {
		cachedOrtValue = default;
		if (!_cache.TryGetValue(file, out InitializerReference? existingReference))
			return false;

		var idx = existingReference.CachedOrtValues.FindIndex(cache => cache.Type == type && ((shape == null && cache.Shape == null) || (shape != null && cache.Shape != null && ((ReadOnlySpan<Int64>)shape).SequenceEqual(cache.Shape))));

		if (idx < 0) 
			return false;
		
		cachedOrtValue = existingReference.CachedOrtValues[idx].Value;
		return true;
	}

	public OrtValue Register(String file, TensorElementType type, Int64[]? shape) {
		if (TryGetCachedValue(file, type, shape, out OrtValue? ortValue)) return ortValue;

		if (!_cache.TryGetValue(file, out InitializerReference? reference)) {
			String initializerPath = Path.Combine(_path, file);
			var temp = File.ReadAllBytes(initializerPath);

			Array.Copy(temp, 0, _initializerBlob, _currentDataOffset, temp.Length);
			var initializerSize = temp.Length;
			reference = new InitializerReference(_currentDataOffset, initializerSize);
			_cache.Add(file, reference);
			Int32 overheadRequiredFor64BitAlignment = initializerSize % 8;
			_currentDataOffset += initializerSize + overheadRequiredFor64BitAlignment;
			Debug.Assert(_currentDataOffset % 8 == 0);
		}

		Debug.Assert(reference.DataOffset % 8 == 0);

		switch (type) {
			case TensorElementType.Int64:
				Int32 int64Offset = reference.DataOffset / sizeof(Int64);
				Int32 int64Length = reference.DataLength / sizeof(Int64);
				ortValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<Int64>(InitializerBlobInt64, int64Offset, int64Length), shape ?? []);
				break;
			case TensorElementType.Float:
				Int32 singleOffset = reference.DataOffset / sizeof(Single);
				Int32 singleLength = reference.DataLength / sizeof(Single);
				ortValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<Single>(InitializerBlobSingle, singleOffset, singleLength), shape ?? []);
				break;
			default:
				throw new NotSupportedException($"Unsupported initializer type: {type}");
		}

		reference.CachedOrtValues.Add(new CachedOrtValue(ortValue, TensorElementType.DataTypeMax, shape));

		return ortValue;
	}

	public void LoadInitializers(string path) {
		if (_initializerBlob.Length > 0) throw new InvalidOperationException("Initializers have already been loaded");
		ArgumentNullException.ThrowIfNull(path);
		if (!Directory.Exists(path)) throw new ArgumentException($"Path {path} does not exist.");
		_path = path;

		EnumerationOptions enumerationOptions = new() { IgnoreInaccessible = true, RecurseSubdirectories = false, AttributesToSkip = FileAttributes.None, MatchType = MatchType.Simple };

		Int64 totalSize = 0;
		foreach (FileInfo file in new DirectoryInfo(path).EnumerateFiles("*", enumerationOptions)) {
			Int64 initializerSize = file.Length;
			Int64 overheadRequiredFor64BitAlignment = initializerSize % 8;
			totalSize += initializerSize + overheadRequiredFor64BitAlignment;
		}

		Debug.Assert(totalSize % 8 == 0);
		if (totalSize > Int32.MaxValue) throw new ArgumentException($"Total size ({totalSize.ToFileSize()}) exceeds maximum of {Int32.MaxValue.ToFileSize()}.");

		_initializerBlob = GC.AllocateArray<Byte>((Int32)totalSize, true);
	}

	#region IDisposable

	/// <inheritdoc />
	public void Dispose() {
		foreach ((_, var value) in _cache) {
			value.Dispose();
		}

		_cache.Clear();
		_initializerBlob = [];
	}

	#endregion
}