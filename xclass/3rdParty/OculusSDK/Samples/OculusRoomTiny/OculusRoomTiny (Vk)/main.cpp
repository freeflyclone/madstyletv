/*****************************************************************************
Filename    :   main.cpp
Content     :   Simple minimal VR demo for Vulkan
Created     :   02/09/2017
Copyright   :   Copyright 2017 Oculus, Inc. All Rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*****************************************************************************/

#define _USE_MATH_DEFINES 
#include "../../OculusRoomTiny_Advanced/Common/Win32_VulkanAppUtil.h"

// VkImage + framebuffer wrapper
class RenderTexture: public VulkanObject
{
public:
    VkImage         image;
    VkImageView     view;
    Framebuffer     fb;

    RenderTexture() :
        image(VK_NULL_HANDLE),
        view(VK_NULL_HANDLE),
        fb()
    {
    }

    bool Create(VkImage anImage, VkFormat format, VkExtent2D size, RenderPass& renderPass, VkImageView depthView)
    {
        image = anImage;

        // Create image view
        VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        CHECKVK(vkCreateImageView(Platform.device, &viewInfo, nullptr, &view));

        CHECK(fb.Create(size, renderPass, view, depthView));

        return true;
    }

    void Release()
    {
        fb.Release();
        if (view) vkDestroyImageView(Platform.device, view, nullptr);
        // Note we don't own image, it will get destroyed when ovr_DestroyTextureSwapChain is called
        image = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
    }
};

// ovrSwapTextureSet wrapper class for Vulkan rendering
class TextureSwapChain: public VulkanObject
{
public:
    ovrSession                  session;
    VkExtent2D                  size;
    ovrTextureSwapChain         textureChain;
    std::vector<RenderTexture>  texElements;

    TextureSwapChain() :
        session(nullptr),
        size{},
        textureChain(nullptr),
        texElements()
    {
    }

    bool Create(ovrSession aSession, VkExtent2D aSize, RenderPass& renderPass, DepthBuffer& depthBuffer)
    {
        session = aSession;
        size = aSize;

        ovrTextureSwapChainDesc desc = {};
        desc.Type = ovrTexture_2D;
        desc.ArraySize = 1;
        desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        desc.Width = (int)size.width;
        desc.Height = (int)size.height;
        desc.MipLevels = 1;
        desc.SampleCount = 1;
        desc.MiscFlags = ovrTextureMisc_DX_Typeless;
        desc.BindFlags = ovrTextureBind_DX_RenderTarget;
        desc.StaticImage = ovrFalse;

        ovrResult result = ovr_CreateTextureSwapChainVk(session, Platform.device, &desc, &textureChain);
        if (!OVR_SUCCESS(result))
            return false;

        int textureCount = 0;
        ovr_GetTextureSwapChainLength(session, textureChain, &textureCount);
        texElements.reserve(textureCount);
        for (int i = 0; i < textureCount; ++i)
        {
            VkImage image;
            result = ovr_GetTextureSwapChainBufferVk(session, textureChain, i, &image);
            texElements.emplace_back(RenderTexture());
            CHECK(texElements.back().Create(image, VK_FORMAT_R8G8B8A8_SRGB, size, renderPass, depthBuffer.view));
        }

        return true;
    }

    const Framebuffer& GetFramebuffer()
    {
        int index = 0;
        ovr_GetTextureSwapChainCurrentIndex(session, textureChain, &index);
        return texElements[index].fb;
    }

    Recti GetViewport()
    {
        return Recti(0, 0, size.width, size.height);
    }

    // Commit changes
    void Commit()
    {
        ovr_CommitTextureSwapChain(session, textureChain);
    }

    void Release()
    {
        if (Platform.device)
        {
            for (auto& te: texElements)
            {
                te.Release();
            }
        }
        if (session && textureChain)
        {
            ovr_DestroyTextureSwapChain(session, textureChain);
        }
        texElements.clear();
        textureChain = nullptr;
        session = nullptr;
    }
};

// ovrMirrorTexture wrapper for rendering the mirror window
class MirrorTexture: public VulkanObject
{
public:
    ovrSession                  session;
    ovrMirrorTexture            mirrorTexture;
    VkImage                     image;

    MirrorTexture() :
        session(nullptr),
        mirrorTexture(nullptr),
        image(VK_NULL_HANDLE)
    {
    }

    bool Create(ovrSession aSession, ovrSizei windowSize)
    {
        session = aSession;

        ovrMirrorTextureDesc mirrorDesc = {};
        mirrorDesc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        mirrorDesc.Width = windowSize.w;
        mirrorDesc.Height = windowSize.h;
        CHECKOVR(ovr_CreateMirrorTextureWithOptionsVk(session, Platform.device, &mirrorDesc, &mirrorTexture));

        CHECKOVR(ovr_GetMirrorTextureBufferVk(session, mirrorTexture, &image));

        return true;
    }

    void Release()
    {
        if (mirrorTexture) ovr_DestroyMirrorTexture(session, mirrorTexture);
        // Note we don't own image, it will get destroyed when ovr_DestroyTextureSwapChain is called
        image = VK_NULL_HANDLE;
        mirrorTexture = nullptr;
        session = nullptr;
    }
};

// return true to retry later (e.g. after display lost)
static bool MainLoop(bool retryCreate)
{
    #define Abort(err) \
    do { \
        retryCreate = false; \
        result = (err); \
        goto Done; \
    } while (0)

    // Per-eye render state
    class EyeState: public VulkanObject
    {
    public:
        ovrSizei                size;
        RenderPass              rp;
        Pipeline                pipe;
        DepthBuffer             depth;
        TextureSwapChain        tex;

        EyeState() :
            size(),
            rp(),
            pipe(),
            depth(),
            tex()
        {
        }

        bool Create(ovrSession session, ovrSizei eyeSize, const PipelineLayout& layout, const ShaderProgram& sp, const VertexBuffer<Vertex>& vb)
        {
            size = eyeSize;
            VkExtent2D vkSize = { (uint32_t)size.w, (uint32_t)size.h };
            CHECK(rp.Create(VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_D32_SFLOAT));
            CHECK(pipe.Create(vkSize, layout, rp, sp, vb));
            CHECK(depth.Create(vkSize, VK_FORMAT_D32_SFLOAT));
            // Note: Format is hard-coded to VK_FORMAT_R8G8B8A8_SRGB till we get a proper ovr_CreateTextureSwapChainVk()
            CHECK(tex.Create(session, vkSize, rp, depth));
            return true;
        }

        void Release()
        {
            tex.Release();
            depth.Release();
            pipe.Release();
            rp.Release();
        }
    };
    
    EyeState                    perEye[ovrEye_Count];

    MirrorTexture               mirrorTexture;

    Scene                       roomScene; 
    bool                        isVisible = true;
    long long                   frameIndex = 0;

    ovrSession                  session;
    ovrGraphicsLuid             luid;

    PipelineLayout              layout;
    ShaderProgram               sp;
    VertexBuffer<Vertex>        vb;

    ovrResult result = ovr_Create(&session, &luid);
    if (!OVR_SUCCESS(result))
        return retryCreate;

    ovrHmdDesc hmdDesc = ovr_GetHmdDesc(session);

    // Setup Window and Graphics
    // Note: the mirror window can be any size, for this sample we use 1/2 the HMD resolution
    ovrSizei windowSize = { hmdDesc.Resolution.w / 2, hmdDesc.Resolution.h / 2 };

    // Initialize the Vulkan renderer
    // Note the ovr_GetSessionPhysicalDeviceVk helper function is called from InitDevice
    if (!Platform.InitDevice("OculusRoomTiny (Vk)", windowSize.w, windowSize.h, session, luid))
    {
        Abort(ovrError_DeviceUnavailable);
    }

    // Begin the initialization command buffer, note that various initialization steps need a valid command buffer to operate correctly
    if (!Platform.CurrentDrawCmd().Begin())
    {
        Debug.Log("Begin command buffer failed");
        Abort(ovrError_InvalidOperation);
    }

    // Create mirror texture
    if (!mirrorTexture.Create(session, windowSize))
    {
        if (retryCreate) goto Done;
        VALIDATE(false, "Failed to create mirror texture.");
    }

    // FloorLevel will give tracking poses where the floor height is 0
    ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);

    if (!sp.Create("vert", "frag"))
    {
        Debug.Log("Failed to create shader program");
        Abort(ovrError_InvalidOperation);
    }

    vb.Attr({ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 })
      .Attr({ 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 })
      .Attr({ 2, 0, VK_FORMAT_R32G32_SFLOAT, 32 });

    if (!layout.Create())
    {
        Debug.Log("Failed to create pipeline layout");
        Abort(ovrError_InvalidOperation);
    }

    // Make a scene
    roomScene.Create(layout, vb);

    // Make per-eye rendering state
    for (auto eye: { ovrEye_Left, ovrEye_Right })
    {
        if (!perEye[eye].Create(session, ovr_GetFovTextureSize(session, eye, hmdDesc.DefaultEyeFov[eye], 1), layout, sp, vb))
        {
            Debug.Log("Failed to create render state for eye " + std::to_string(eye));
            Abort(ovrError_InvalidOperation);
        }
    }

    // Get swapchain images ready for blitting (use drawCmd instead of xferCmd to keep things simple)
    Platform.sc.Prepare(Platform.CurrentDrawCmd().buf);

    // Perform all init-time commands
    if (!(Platform.CurrentDrawCmd().End() && Platform.CurrentDrawCmd().Exec(Platform.drawQueue) && Platform.CurrentDrawCmd().Wait()))
    {
        Debug.Log("Executing initial command buffer failed");
        Abort(ovrError_InvalidOperation);
    }
    Platform.CurrentDrawCmd().Reset();

    // Let the compositor know which queue to synchronize with
    ovr_SetSynchonizationQueueVk(session, Platform.drawQueue);

    Debug.Log("Main loop...");

    // Main loop
    while (Platform.HandleMessages())
    {
        // Keyboard inputs to adjust player orientation
        static float yaw(3.141592f); // I like pie
        if (Platform.Key[VK_LEFT])  yaw += 0.02f;
        if (Platform.Key[VK_RIGHT]) yaw -= 0.02f;

        // Keyboard inputs to adjust player position
        static Vector3f playerPos(0.0f, 0.0f, -5.0f);
        if (Platform.Key['W'] || Platform.Key[VK_UP])   playerPos += Matrix4f::RotationY(yaw).Transform(Vector3f( 0.00f, 0, -0.05f));
        if (Platform.Key['S'] || Platform.Key[VK_DOWN]) playerPos += Matrix4f::RotationY(yaw).Transform(Vector3f( 0.00f, 0, +0.05f));
        if (Platform.Key['D'])                          playerPos += Matrix4f::RotationY(yaw).Transform(Vector3f(+0.05f, 0,  0.00f));
        if (Platform.Key['A'])                          playerPos += Matrix4f::RotationY(yaw).Transform(Vector3f(-0.05f, 0,  0.00f));

        Matrix4f rollPitchYaw = Matrix4f::RotationY(yaw);
        
        // Animate the cube
        static float cubeClock = 0;
        roomScene.models[0].pos = Vector3f(9 * sinf(cubeClock), 3, 9 * cosf(cubeClock += 0.015f));
        if (cubeClock >= 2.0f * float(M_PI))
            cubeClock -= 2.0f * float(M_PI);

        // Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. hmdToEyePose) may change at runtime.
	      ovrEyeRenderDesc eyeRenderDesc[ovrEye_Count];
        for (auto eye: { ovrEye_Left, ovrEye_Right })
            eyeRenderDesc[eye]  = ovr_GetRenderDesc(session, eye,  hmdDesc.DefaultEyeFov[eye]);

        // Get eye poses, feeding in correct IPD offset
        ovrPosef HmdToEyePose[ovrEye_Count] = { eyeRenderDesc[ovrEye_Left].HmdToEyePose,
                                                eyeRenderDesc[ovrEye_Right].HmdToEyePose};
        ovrPosef eyeRenderPose[ovrEye_Count];
        double sensorSampleTime; // sensorSampleTime is fed into ovr_SubmitFrame later
        ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyePose, eyeRenderPose, &sensorSampleTime);

        if (isVisible)
        {
            Platform.NextDrawCmd();
            auto& cmd = Platform.CurrentDrawCmd();
            cmd.Reset();
            cmd.Begin();

            for (auto eye: { ovrEye_Left, ovrEye_Right })
            {
                // Switch to eye render target
                static std::array<VkClearValue, 2> clearValues;
                clearValues[0].color.float32[0] = 0.0f;
                clearValues[0].color.float32[1] = 0.0f;
                clearValues[0].color.float32[2] = 0.0f;
                clearValues[0].color.float32[3] = 1.0f;
                clearValues[1].depthStencil.depth = 1.0f;
                clearValues[1].depthStencil.stencil = 0;
                VkRenderPassBeginInfo rpBegin = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
                rpBegin.renderPass = perEye[eye].rp.pass;
                rpBegin.framebuffer = perEye[eye].tex.GetFramebuffer().fb;
                rpBegin.renderArea = { { 0, 0 }, perEye[eye].tex.size };
                rpBegin.clearValueCount = (uint32_t)clearValues.size();
                rpBegin.pClearValues = clearValues.data();

                vkCmdBeginRenderPass(cmd.buf, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

                vkCmdBindPipeline(cmd.buf, VK_PIPELINE_BIND_POINT_GRAPHICS, perEye[eye].pipe.pipe);

                // Get view and projection matrices
                Matrix4f finalRollPitchYaw = rollPitchYaw * Matrix4f(eyeRenderPose[eye].Orientation);
                Vector3f finalUp = finalRollPitchYaw.Transform(Vector3f(0, 1, 0));
                Vector3f finalForward = finalRollPitchYaw.Transform(Vector3f(0, 0, -1));
                Vector3f shiftedEyePos = playerPos + rollPitchYaw.Transform(eyeRenderPose[eye].Position);

                Matrix4f view = Matrix4f::LookAtRH(shiftedEyePos, shiftedEyePos + finalForward, finalUp);
                Matrix4f proj = ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_None);

                // Render world
                roomScene.Render(view, proj, layout, vb);

                vkCmdEndRenderPass(cmd.buf);
            }

            cmd.End();
            cmd.Exec(Platform.drawQueue);

            // Commit changes to the textures so they get picked up by the compositor
            for (auto eye: { ovrEye_Left, ovrEye_Right })
            {
                perEye[eye].tex.Commit();
            }
        }
        else // Sleep to avoid spinning on mirror updates while HMD is doffed
        {
            ::Sleep(10);
        }

        // Submit rendered eyes as an EyeFov layer
        ovrLayerEyeFov ld;
        ld.Header.Type  = ovrLayerType_EyeFov;
        ld.Header.Flags = 0;
        ld.SensorSampleTime  = sensorSampleTime;
        for (auto eye: { ovrEye_Left, ovrEye_Right })
        {
            ld.ColorTexture[eye] = perEye[eye].tex.textureChain;
            ld.Viewport[eye]     = perEye[eye].tex.GetViewport();
            ld.Fov[eye]          = hmdDesc.DefaultEyeFov[eye];
            ld.RenderPose[eye]   = eyeRenderPose[eye];
        }

        ovrLayerHeader* layers = &ld.Header;
        ovrResult result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);
        // Exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
        if (!OVR_SUCCESS(result))
            goto Done;

        isVisible = (result == ovrSuccess);

        ovrSessionStatus sessionStatus;
        ovr_GetSessionStatus(session, &sessionStatus);
        if (sessionStatus.ShouldQuit)
            goto Done;
        if (sessionStatus.ShouldRecenter)
            ovr_RecenterTrackingOrigin(session);

        // Blit mirror texture to the swapchain's back buffer
        // For now block until we have an output to render into
        // The swapchain uses VK_PRESENT_MODE_IMMEDIATE_KHR or VK_PRESENT_MODE_MAILBOX_KHR to avoid blocking eye rendering
        Platform.sc.Aquire();

        Platform.xferCmd.Reset();
        Platform.xferCmd.Begin();

        auto presentImage = Platform.sc.image[Platform.sc.renderImageIdx];

        // PRESENT_SRC_KHR -> TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier presentBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        presentBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        presentBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.image = presentImage;
        presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentBarrier.subresourceRange.baseMipLevel = 0;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.baseArrayLayer = 0;
        presentBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(Platform.xferCmd.buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &presentBarrier);

        // Blit
        VkImageBlit region = {};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.mipLevel = 0;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[0] = { 0, 0, 0 };
        region.srcOffsets[1] = { windowSize.w, windowSize.h, 1 };
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.mipLevel = 0;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[0] = { 0, 0, 0 };
        region.dstOffsets[1] = { windowSize.w, windowSize.h, 1 };
        vkCmdBlitImage(Platform.xferCmd.buf, mirrorTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            presentImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

        // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR
        presentBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        presentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        presentBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.image = presentImage;
        presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentBarrier.subresourceRange.baseMipLevel = 0;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.baseArrayLayer = 0;
        presentBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(Platform.xferCmd.buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &presentBarrier);

        Platform.xferCmd.End();

        Platform.xferCmd.Exec(Platform.xferQueue);
        Platform.xferCmd.Wait();

        // For now just block on Aquire's fence, could use a semaphore with e.g.:
        // Platform.sc.Present(Platform.xferQueue, Platform.xferDone);
        Platform.sc.Present(Platform.xferQueue, VK_NULL_HANDLE);

        ++frameIndex;
    }

Done:
    Debug.Log("Exiting main loop...");

    roomScene.Release();

    vb.Release();
    sp.Release();
    layout.Release();

    mirrorTexture.Release();

    for (auto eye: { ovrEye_Left, ovrEye_Right })
    {
        perEye[eye].Release();
    }

    Platform.ReleaseDevice();
    ovr_Destroy(session);

    // Retry on ovrError_DisplayLost
    return retryCreate || OVR_SUCCESS(result) || (result == ovrError_DisplayLost);
}

//-------------------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR, int)
{
    // Initializes LibOVR, and the Rift
    ovrResult result = ovr_Initialize(nullptr);
    VALIDATE(OVR_SUCCESS(result), "Failed to initialize libOVR.");

    VALIDATE(Platform.InitWindow(hinst, L"Oculus Room Tiny (Vulkan)"), "Failed to open window.");

    Platform.Run(MainLoop);

    ovr_Shutdown();

    return(0);
}
