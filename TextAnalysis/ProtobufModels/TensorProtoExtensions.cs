namespace TextAnalysis.ProtobufModels;

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Onnx;

internal static class TensorProtoExtensions {
	public static ReadOnlySpan<Byte> GetRawData(this TensorProto tensor) {
		if (tensor.RawData != null) return tensor.RawData;
		
		if (tensor.Int32Datas != null) return  MemoryMarshal.Cast<Int32, Byte>(new ReadOnlySpan<Int32>(tensor.Int32Datas));
		if (tensor.Int64Datas != null) return  MemoryMarshal.Cast<Int64, Byte>(new ReadOnlySpan<Int64>(tensor.Int64Datas));
		if (tensor.Uint64Datas != null) return  MemoryMarshal.Cast<UInt64, Byte>(new ReadOnlySpan<UInt64>(tensor.Uint64Datas));
		if (tensor.FloatDatas != null) return  MemoryMarshal.Cast<Single, Byte>(new ReadOnlySpan<Single>(tensor.FloatDatas));
		if (tensor.DoubleDatas != null) return  MemoryMarshal.Cast<Double, Byte>(new ReadOnlySpan<Double>(tensor.DoubleDatas));
		
		throw new InvalidOperationException($"Unable to get raw Byte[] data from {tensor.Name}@{(TensorProto.DataType)tensor.data_type}");
	}
	
	public static Single[] GetSingleData(this TensorProto tensor) {
		if (tensor.FloatDatas != null && tensor.FloatDatas.Length > 0) return tensor.FloatDatas;
		if (tensor.DoubleDatas != null && tensor.DoubleDatas.Length > 0) return Array.ConvertAll(tensor.DoubleDatas, l => (Single)l);
		if (tensor.Int32Datas != null && tensor.Int32Datas.Length > 0) return Array.ConvertAll(tensor.Int32Datas, l => (Single)l);
		if (tensor.Int64Datas != null && tensor.Int64Datas.Length > 0) return Array.ConvertAll(tensor.Int64Datas, l => (Single)l);
		if (tensor.Uint64Datas != null && tensor.Uint64Datas.Length > 0) return Array.ConvertAll(tensor.Uint64Datas, l => (Single)l);
		if (tensor.RawData != null && tensor.RawData.Length > 0) {
			return Enumerable
				.Range(0, tensor.RawData.Length / 4)
				.Select(idx => BitConverter.ToSingle(tensor.RawData, idx * 4))
				.ToArray();
		}

		throw new InvalidOperationException($"Unable to get Single[] data from {tensor.Name}@{(TensorProto.DataType)tensor.data_type}");
	}
	
	public static Int64[] GetIntData(this TensorProto tensor) {
		if (tensor.Int64Datas != null && tensor.Int64Datas.Length > 0) return tensor.Int64Datas;
		if (tensor.Int32Datas != null && tensor.Int32Datas.Length > 0) return Array.ConvertAll(tensor.Int32Datas, l => (Int64)l);

		throw new InvalidOperationException($"Unable to get Int64[] data from {tensor.Name}@{(TensorProto.DataType)tensor.data_type}");
	}

	public static Int64 CalculateSize(this TensorProto tensor) {
		if (tensor.RawData != null) return tensor.RawData.Length;
		if (tensor.Int64Datas != null) return tensor.Int64Datas.Length * sizeof(Int64);
		if (tensor.Int32Datas != null) return tensor.Int32Datas.Length * sizeof(Int32);
		if (tensor.DoubleDatas != null) return tensor.DoubleDatas.Length * sizeof(Double);
		if (tensor.FloatDatas != null) return tensor.FloatDatas.Length * sizeof(Single);
		if (tensor.StringDatas != null) return tensor.StringDatas.Sum(sd => sd.Length) * sizeof(Char);

		throw new InvalidOperationException($"Unable to determine size of tensor: {(TensorProto.DataType)tensor.data_type}");
	}
}