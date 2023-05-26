module;

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

export module Buffer;


export class Buffer {
public:
    Buffer(VmaAllocator vmaAllocator, vk::BufferUsageFlags usageFlags, size_t size)
        : allocator(vmaAllocator), size(size) {
        vk::BufferCreateInfo bufferInfo{
                .size = size,
                .usage = usageFlags | vk::BufferUsageFlagBits::eTransferDst,
        };

        VkBufferCreateInfo buffInfo = static_cast<VkBufferCreateInfo>(bufferInfo);

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        VkBuffer buff;
        if (vmaCreateBuffer(vmaAllocator, &buffInfo, &allocInfo, &buff, &allocation, &allocationInfo) != VK_SUCCESS)
            throw std::runtime_error("could not allocate buffer");
        buffer = buff;
    }

    ~Buffer() {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }

    size_t getSize() const {
        return size;
    }

    const vk::Buffer& getHandle() const {
        return buffer;
    }

private:
    size_t size;
    VmaAllocator allocator;
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    vk::Buffer buffer;

};
