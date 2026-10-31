CXX = g++
CXXFLAGS = -std=c++20 -Wall -I$(MATRIXLIBRARY)
COMMON_HEADERS = single_network_layer.hpp input_output.hpp network_operations.hpp

all: test_network_output test_gradients train_network

test_network_output: test_network_output.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

test_gradients: test_gradients.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

train_network: train_network.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f test_network_output test_gradients train_network
