#include <iostream>
#include <fstream>
#include "calibrator.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << msg << std::endl;
		}
	}
}logger;

void ONNX2TensorRT(const char* ONNX_file, std::string& Engine_file, bool& FP16, bool& INT8, std::string& image_dir, const char*& calib_table) {
	std::cout << "Load ONNX file form: " << ONNX_file << "\nStart export..." << std::endl;
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

	parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		std::cout << parser->getError(i)->desc() << std::endl;

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
	if (FP16) {
		if (!builder->platformHasFastFp16()) {
			std::cout << "不支持FP16量化！" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else if (INT8) {
		if (!builder->platformHasFastInt8()) {
			std::cout << "不支持INT8量化！" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		nvinfer1::IInt8EntropyCalibrator2* calibrator = new Calibrator(1, 640, 640, image_dir, calib_table);
		config->setInt8Calibrator(calibrator);
	}

	nvinfer1::IHostMemory* serializeModel = builder->buildSerializedNetwork(*network, *config);

	std::ofstream engine(Engine_file, std::ios::binary);
	engine.write(reinterpret_cast<const char*>(serializeModel->data()), serializeModel->size());

	delete parser;
	delete network;
	delete config;
	delete builder;

	delete serializeModel;
	std::cout << "Export success, Save as: " << Engine_file << std::endl;
}

int main(int argc, char** argv) {
	const char* ONNX_file = "../weights/yolov8n.onnx";
	std::string Engine_file = "../weights/yolov8n.engine";

	std::string image_dir = "../images/";
	const char* calib_table = "../weights/calibrator.table";

	bool FP16 = true;
	bool INT8 = false;

	std::ifstream file(ONNX_file, std::ios::binary);
	if (!file.good()) {
		std::cout << "Load ONNX file failed！" << std::endl;
	}

	ONNX2TensorRT(ONNX_file, Engine_file, FP16, INT8, image_dir, calib_table);

	return 0;
}
