module;

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

export module StreamBuffer;

export class StreamBuffer {
public:
	StreamBuffer(VmaAllocator allocator, vk::BufferUsageFlags usageFlags, size_t size)
		: allocator(allocator), size(size) {
		vk::BufferCreateInfo bufferInfo{
				.size = size,
				.usage = usageFlags,
		};

		VkBufferCreateInfo buffInfo = static_cast<VkBufferCreateInfo>(bufferInfo);

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

		VkBuffer buff;
		if (vmaCreateBuffer(allocator, &buffInfo, &allocInfo, &buff, &allocation, &allocationInfo) != VK_SUCCESS)
			throw std::runtime_error("could not allocate buffer");
		buffer = buff;
	}

	~StreamBuffer() {
		vmaDestroyBuffer(allocator, buffer, allocation);
	}

	void* getMappedData() {
		return allocationInfo.pMappedData;
	}

	vk::Buffer getHandle() const {
		return buffer;
	}

private:
	size_t size;
	VmaAllocator allocator;
	VmaAllocation allocation;
	VmaAllocationInfo allocationInfo;
	vk::Buffer buffer;

};
