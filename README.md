# Paper List | High Dynamic Range Imaging

This repository compiles a list of papers related to HDR. (Updating)

Continual improvements are being made to this repository. If you come across any relevant papers that should be included, please don't hesitate to open an issue.

## Contents

- [Papers](#hdr-papers)
  - [Multi-Frame HDRI](#multi-frame-hdri)
  - [Single-Frame HDRI](#single-frame-hdri)
  - [HDRTV](#hdrtv)
  - [HDR Video](#hdr-video)
  - [Tone Mapping](#tone-mapping)
  - [Traditional HDRI](#traditional-hdri)
  - [Other](#other)
- [Challenges](#hdr-challenges)
- [Datasets](#hdr-datasets)
  - [HDR Image Datasets](#hdr-image-datasets)
  - [HDR Video Datasets](#hdr-video-datasets)

## HDR Papers

### Multi-Frame HDRI

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm | [ICCV-2025](https://arxiv.org/pdf/2506.23537) | [AFUNet]() |     |     |
| From Dynamic to Static: Stepwisely Generate HDR Image for Ghost Removal | [TCSVT-2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10693592) |     |     |     |
| UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion | [CVPR-2025](https://arxiv.org/abs/2501.11515) | [UltraFusion](https://github.com/OpenImagingLab/UltraFusion) |     |     |
| SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging | [ECCV-2024](https://www.arxiv.org/pdf/2407.16308) | [SAFNet](https://github.com/ltkong218/SAFNet) |     |     |
| PASTA: Towards Flexible and Efficient HDR Imaging Via Progressively Aggregated Spatio-Temporal Alignment | [arxiv-2024](https://arxiv.org/pdf/2403.10376) |     |     |     |
| Enhancing Multi-Exposure High Dynamic Range Imaging with Overlapped Codebook for Improved Representation Learning | [ICPR-2024](https://arxiv.org/pdf/2507.01588) | --- | --- | --- |
| Generating Content for HDR Deghosting from Frequency View | [CVPR-2024](https://arxiv.org/pdf/2404.00849) |     |     |     |
| Self-Supervised High Dynamic Range Imaging with Multi-Exposure Images in Dynamic Scenes | [ICLR-2024](https://arxiv.org/pdf/2310.01840.pdf) | [selfHDR](https://github.com/cszhilu1998/SelfHDR) |     | Few shot |
| Joint Denoising and Fusion with Short- and Long-exposure Raw Pairs | [arxiv-2023](https://arxiv.org/pdf/2306.10311.pdf) |     |     |     |
| Alignment-free HDR Deghosting with Semantics Consistent Transformer | [ICCV-2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Tel_Alignment-free_HDR_Deghosting_with_Semantics_Consistent_Transformer_ICCV_2023_paper.pdf) | [SCTNet](https://github.com/Zongwei97/SCTNet) | [ICCV23](https://github.com/Zongwei97/SCTNet) | New dataset |
| Towards High-quality HDR Deghosting with Conditional Diffusion Models | [TCSVT-2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10288540) |     |     |     |
| A Unified HDR Imaging Method with Pixel and Patch Level | [CVPR-2023](https://arxiv.org/pdf/2304.06943.pdf) |     |     |     |
| HDR Imaging with Spatially Varying Signal-to-Noise Ratios | [CVPR-2023](https://arxiv.org/pdf/2303.17253.pdf) |     |     |     |
| SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders | [CVPR-2023](https://arxiv.org/pdf/2304.06914.pdf) |     | Kalantari, Hu | Few shot HDR |
| Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset | [CVPR-2023](https://web.comp.polyu.edu.hk/cslzhang/paper/CVPR23-Joint-HDRDN.pdf) | [Joint-HDRDN](https://github.com/shuaizhengliu/Joint-HDRDN) | Mobile-HDR | New mobile HDR dataset, Transformer-based model |
| Improving Dynamic HDR Imaging with Fusion Transformer | [AAAI-2023](https://ojs.aaai.org/index.php/AAAI/article/view/25107) |     |     |     |
| Robust Real-world Image Enhancement Based on Multi-Exposure LDR Images | [WACV-2023](https://openaccess.thecvf.com/content/WACV2023/html/Ren_Robust_Real-World_Image_Enhancement_Based_on_Multi-Exposure_LDR_Images_WACV_2023_paper.html) |     | Kalantari&Various | Cost volumn |
| SJ-HD2R: Selective Joint High Dynamic Range and Denoising Imaging for Dynamic Scenes | [arxiv-2022](https://arxiv.org/pdf/2206.09611.pdf) |     |     |     |
| FlexHDR: Modelling Alignment and Exposure Uncertainties for Flexible HDR Imaging | [TIP-2022](https://arxiv.org/abs/2201.02625) |     | Kalantari | Arbitrary number of input LDRs, HDR flow |
| **Selective TransHDR: Transformer-based selective HDR Imaging using Ghost Region Mask** | [ECCV-2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770292.pdf) |     | Kalantari | Ghost Region Mask |
| **Ghost-free High Dynamic Range Imaging with Context-aware Transformer** | [ECCV-2022](https://arxiv.org/pdf/2208.05114) | [HDR-Transformer](https://github.com/megvii-research/HDR-Transformer) | Kalantari | Context-Aware Transformer |
| A Lightweight Network for High Dynamic Range Imaging | [CVPRW-2022](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yan_A_Lightweight_Network_for_High_Dynamic_Range_Imaging_CVPRW_2022_paper.pdf) |     | NTIRE | Two branch, Lightweight |
| Gamma-Enhanced Spatial Attention Network for Efficient High Dynamic Range Imaging | [CVPRW-2022](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Li_Gamma-Enhanced_Spatial_Attention_Network_for_Efficient_High_Dynamic_Range_Imaging_CVPRW_2022_paper.pdf) |     | NTIRE | gamma-corrected |
| Bidirectional Motion Estimation With Cyclic Cost Volume for High Dynamic Range Imaging | [CVPRW-2022](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Vien_Bidirectional_Motion_Estimation_With_Cyclic_Cost_Volume_for_High_Dynamic_CVPRW_2022_paper.pdf) |     | NTIRE | Bidirectional Motion Estimation |
| Attention-Guided Progressive Neural Texture Fusion for High Dynamic Range Image Restoration | [TIP-2022](https://arxiv.org/pdf/2107.06211.pdf) |     | Kalnatari | Two stream, Neural Feature Transfer |
| Dual-Attention-Guided Network for Ghost-Free High Dynamic Range Imaging | [IJCV-2022](https://link.springer.com/article/10.1007/s11263-021-01535-y) |     | Kalantari |     |
| High Dynamic Range Imaging of Dynamic Scenes with Saturation Compensation but without Explicit Motion Compensation | [WACV-2022](https://openaccess.thecvf.com/content/WACV2022/supplemental/Chung_High_Dynamic_Range_WACV_2022_supplemental.pdf) | [hdri-saturation-compensation](https://github.com/haesoochung/hdri-saturation-compensation) | Kalantari | Brightness adjustment, Saturation mask |
| Drbr-Hdr: Dual-Branch Recursive Band Reconstruction Network for Hdr with Large Motions | [SSRN-2022](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4147499) |     | Kalantari | Dual branch |
| Learning Regularized Multi-Scale Feature Flow for High Dynamic Range Imaging | [arxiv-2022](https://arxiv.org/pdf/2207.02539.pdf) |     | Kalantari | Multi-scale, Flow |
| High Dynamic Range Imaging via Gradient-aware Context Aggregation Network | [PR-2022](https://www.sciencedirect.com/science/article/pii/S0031320321005227) |     | Kalantari | Gradient information |
| Labeled from Unlabeled: Exploiting Unlabeled Data for Few-shot Deep HDR Deghosting | [CVPR-2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Prabhakar_Labeled_From_Unlabeled_Exploiting_Unlabeled_Data_for_Few-Shot_Deep_HDR_CVPR_2021_paper.pdf) | [FSHDR](https://github.com/Susmit-A/FSHDR) | Kalantari | Few shot |
| **HDR-GAN: HDR Image Reconstruction from Multi-Exposed LDR Images with Large Motions** | [TIP-2021](https://arxiv.org/abs/2007.01628) | [HDR-GAN](https://github.com/nonu116/HDR-GAN) | Kalantari | GAN |
| Ghost-Free Deep High-Dynamic-Range Imaging Using Focus Pixels for Complex Motion Scenes | [TIP-2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9429936) |     | own real-world dataset | Utilize focus pixel image |
| Self-Gated Memory Recurrent Network for Efficient Scalable HDR Deghosting | [TCI-2021](https://ieeexplore.ieee.org/document/9540317) | [HDRRNN](https://github.com/Susmit-A/HDRRNN) | Kalantari + Prabhakar | Recurrent |
| ADNet: Attention-guided Deformable Convolutional Network for High Dynamic Range Imaging | [CVPRW-2021](https://arxiv.org/pdf/2105.10697.pdf) | [ADNet](https://github.com/liuzhen03/ADNet) | NTIRE | PCD |
| Progressive and Selective Fusion Network for High Dynamic Range Imaging | [MM-2021](https://arxiv.org/pdf/2108.08585.pdf) |     | Kalantari | Progressive fusion |
| Merging-ISP: Multi-Exposure High Dynamic Range Image Signal Processing | [GCPR-2021](https://link.springer.com/chapter/10.1007/978-3-030-92659-5_21) |     | Kalantari | ISP |
| Towards Accurate HDR Imaging with Learning Generator Constraints | [Neurocomputing-2021](https://www.sciencedirect.com/science/article/pii/S092523122031849X) |     | Kalantari | LDR-->HDR-->LDR |
| Hierarchical Fusion for Practical Ghost-free High Dynamic Range Imaging | [MM-2021](https://dl.acm.org/doi/abs/10.1145/3474085.3475260) |     | Kalantari | Hierarchical fusion |
| **Deep HDR Imaging via A Non-Local Network** | [TIP-2020](https://ieeexplore.ieee.org/abstract/document/8989959) | [NHDRRNet (Keras-implementation)](https://github.com/tuvovan/NHDRRNet), [NHDRRNet (PyTorch-re-implementation)](https://github.com/Galaxies99/NHDRRNet-pytorch) | Kalantari | Non-local |
| Towards Practical and Efficient High-resolution HDR Deghosting with CNN | [ECCV-2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660494.pdf) |     | Kalantari | Efficient |
| Sensor-realistic Synthetic Data Engine for Multi-frame High Dynamic Range Photography | [CVPRW-2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Hu_Sensor-Realistic_Synthetic_Data_Engine_for_Multi-Frame_High_Dynamic_Range_Photography_CVPRW_2020_paper.pdf) | [code](https://github.com/nadir-zeeshan/sensor-realistic-synthetic-data) | synthetic data | synthetic data from game engine |
| Ghost Removal via Channel Attention in Exposure Fusion | [CVIU-2020](https://reader.elsevier.com/reader/sd/pii/S1077314220301132?token=988EBE214F2D779B43AFBD9865B305D565601632AE5B46780AB792995116FC17FE7FBFCE6C45ECE689F9F7BB11ACCA83&originRegion=us-east-1&originCreation=20221126071114) |     | Kalantari | Non-local |
| Pyramid inter-attention for high dynamic range imaging | [Sensors-2020](https://www.mdpi.com/1424-8220/20/18/5102) |     | Kalantari | Attention |
| Attention-Mask Dense Merger (Attendense) Deep HDR for Ghost Removal | [ICASSP-2020](http://www.personal.psu.edu/kmm1122/Publications/Attendense.pdf) |     | Kalantari | Attention Mask |
| Exposure-Structure Blending Network for High Dynamic Range Imaging of Dynamic Scenes github\ | [Access-2020](https://ieeexplore.ieee.org/document/9125884) | [ESBN](https://github.com/tkd1088/ESBN) | Kalantari | Encoder-decoder |
| Multi-scale Dense Networks for Deep High Dynamic Range Imaging | [WACV-2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658831&tag=1) |     | Kalantari | Multi-scale, DenseUnet |
| **Attention-guided Network for Ghost-free High Dynamic Range Imaging** | [CVPR-2019](https://arxiv.org/abs/1904.10293) | [AHDRNet](https://github.com/qingsenyangit/AHDRNet) | Kalantari | Channel Attention |
| Kernel Prediction Network for Detail-Preserving High Dynamic Range Imaging | [APSIPA-ASC-2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9023217) |     | Kalantari | Kernal prediction |
| Deep Multi-Stage Learning for HDR With Large Object Motions | [ICIP-2019](https://ieeexplore.ieee.org/document/8803582) |     | Kalantari | Generate virtual exposures |
| A Fast, Scalable, and Reliable Deghosting Method for Extreme Exposure Fusion | [ICCP-2019](https://ieeexplore.ieee.org/document/8747329) | [Deep Deghosting HDR](https://github.com/rajat95/Deep-Deghosting-HDR) | Prabhakar | Arbitrary number of inputs |
| **Deep High Dynamic Range Imaging with Large Foreground Motions** | [ECCV-2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangzhe_Wu_Deep_High_Dynamic_ECCV_2018_paper.pdf) | [DeepHDR](https://github.com/elliottwu/DeepHDR) | Kalantari | U-Net |
| Deep HDR Reconstruction of Dynamic Scenes | [ICIVC 2018](https://ieeexplore.ieee.org/abstract/document/8492856) |     | Kalantari+own | FlowNet2.0 |
| **Deep high dynamic range imaging of dynamic scenes** | [SIGGRAPH-2017](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR.pdf) | [Kalantari (Official MATLAB implementation)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Code_v1.0.zip), [TensorFlow implementation](https://github.com/TH3CHARLie/deep-high-dynamic-range) | Kalantari | Fisrt deep multi-frame HDRI |

### Single-Frame HDRI

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| LEDiff: Latent Exposure Diffusion for HDR Generation | [CVPR-2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_LEDiff_Latent_Exposure_Diffusion_for_HDR_Generation_CVPR_2025_paper.pdf) | [LEDiff]() |     |     |
| DCDR-UNet: Deformable Convolution Based Detail Restoration via U-shape Network for Single Image HDR Reconstruction | [CVPRW-2024](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Kim_DCDR-UNet_Deformable_Convolution_Based_Detail_Restoration_via_U-shape_Network_for_CVPRW_2024_paper.pdf) |     |     |     |
| Single-Image HDR Reconstruction by Multi-Exposure Generation | [WACV-2023](https://openaccess.thecvf.com/content/WACV2023/papers/Le_Single-Image_HDR_Reconstruction_by_Multi-Exposure_Generation_WACV_2023_paper.pdf) | [code](https://github.com/VinAIResearch/single_image_hdr) | DrTMO |     |
| Single image ldr to hdr conversion using conditional diffusion | [ICIP-2023](https://arxiv.org/pdf/2307.02814) |     |     |     |
| RawHDR: High Dynamic Range Image Reconstruction from a Single Raw Image | [ICCV-2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zou_RawHDR_High_Dynamic_Range_Image_Reconstruction_from_a_Single_Raw_ICCV_2023_paper.pdf) |     | [RawHDR](https://github.com/jackzou233/RawHDR) |     |
| LHDR: HDR Reconstruction for Legacy Content using a Lightweight DNN | [ACCV-2022](https://openaccess.thecvf.com/content/ACCV2022/papers/Guo_LHDR_HDR_Reconstruction_for_Legacy_Content_using_a_Lightweight_DNN_ACCV_2022_paper.pdf) | /   |     |     |
| Ultra-High-Definition Image HDR Reconstruction via Collaborative Bilateral Learning | [ICCV-2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Ultra-High-Definition_Image_HDR_Reconstruction_via_Collaborative_Bilateral_Learning_ICCV_2021_paper.html) |     |     |     |
| A Two-stage Deep Network for High Dynamic Range Image Reconstruction | [CVPRW-2021](https://arxiv.org/pdf/2104.09386.pdf) | [code](https://github.com/sharif-apu/twostageHDR_NTIRE21) | NTIRE 2021 |     |
| HDRUNet: Single Image HDR Reconstruction With Denoising and Dequantization | [CVPRW-2021](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_HDRUNet_Single_Image_HDR_Reconstruction_With_Denoising_and_Dequantization_CVPRW_2021_paper.pdf) | [HDRUNet](https://github.com/chxy95/HDRUNet) | NTIRE 2021 |     |
| Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline | [CVPR-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline_CVPR_2020_paper.pdf) | [SingleHDR](https://github.com/alex04072000/SingleHDR) |     |     |
| Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss | [SIGGRAPH 2020](https://arxiv.org/abs/2005.07335) | [HDRCNN](https://github.com/marcelsan/Deep-HdrReconstruction) |     |     |
| *ExpandNet: A Deep Convolutional Neural Network for High Dynamic Range Expansion from Low Dynamic Range Content* | [Eurographics-2018](https://arxiv.org/pdf/1803.02266) | [ExpandNet](https://github.com/dmarnerides/hdr-expandnet) |     |     |
| Deep Reverse Tone Mapping | [TOG-2017](https://dl.acm.org/doi/10.1145/3130800.3130834) |     |     |     |

### HDRTV

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models | [CVPR-2023](https://arxiv.org/pdf/2303.13031.pdf) | [HDRTVDM](https://github.com/AndreGuo/HDRTVDM) |     |     |
|     |     |     |     |     |

### HDR Video

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| Exposure Completing for Temporally Consistent Neural High Dynamic Range Video Rendering | [ACMMM-2024](https://arxiv.org/pdf/2407.13309) | [NECHDR](https://github.com/JiahaoCui-HUST/NECHDR) |     |     |
| HDRFlow: Real-Time HDR Video Reconstruction with Large Motions | [CVPR-2024](https://arxiv.org/abs/2403.03447) | [HDRFlow](https://github.com/OpenImagingLab/HDRFlow) |     |     |
| Towards Real-World HDR Video Reconstruction: A Large-Scale Benchmark Dataset and A Two-Stage Alignment Network | [CVPR-2024](https://arxiv.org/pdf/2405.00244) | [Real-HDRV](https://github.com/yungsyu99/Real-HDRV) |     |     |
| Self-Supervised High Dynamic Range Imaging: What Can Be Learned from a Single 8-bit Video? | [TOG-2024](https://dl.acm.org/doi/pdf/10.1145/3648570) |  [Zeroshot-HDRV](https://github.com/banterle/Zeroshot-HDRV)   |     |     |
| HDR Video Reconstruction with a Large Dynamic Dataset in Raw and sRGB Domavidins | [arxiv-2023](https://arxiv.org/pdf/2304.04773.pdf) | /   |     |     |
| Bidirectional Translation Between UHD-HDR and HD-SDR Videos | [TMM-2023](https://ieeexplore.ieee.org/abstract/document/10025794?casa_token=7DDZ-maF_TsAAAAA:hrzFZojojKx716_gEkJUv9_91i1GmP_-r-OnBnWTet-F11USwricACjJj7PCCe8AOw6fpZp0) | [HDR-BiTNet](https://github.com/mdyao/HDR-BiTNet) |     |     |
| 1000 FPS HDR Video with a Spike-RGB Hybrid Camera | [CVPR-2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chang_1000_FPS_HDR_Video_With_a_Spike-RGB_Hybrid_Camera_CVPR_2023_paper.html) |     |     |     |
| LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction | [ICCV-2023](https://openaccess.thecvf.com/content/ICCV2023/html/Chung_LAN-HDR_Luminance-based_Alignment_Network_for_High_Dynamic_Range_Video_Reconstruction_ICCV_2023_paper.html) | [LAN-HDR](https://github.com/haesoochung/LAN-HDR) |     |     |
| Learning Event Guided High Dynamic Range Video Reconstruction | [CVPR-2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Learning_Event_Guided_High_Dynamic_Range_Video_Reconstruction_CVPR_2023_paper.html) | [HDRev](https://github.com/YixinYang-00/HDRev) |     |     |
| HDR Video Reconstruction: A Coarse-to-fine Network and A Real-world Benchmark Dataset | [ICCV-2021](https://arxiv.org/abs/2103.14943) | [DeepHDRVideo](https://github.com/guanyingc/DeepHDRVideo) |     | Coarse to fine, real world dataset |
| Patch-Based High Dynamic Range Video | [SigAisa-13](https://web.ece.ucsb.edu/~psen/Papers/SIGASIA13_HDRVideo_LoRes.pdf) | [Kalantari13](https://web.ece.ucsb.edu/~psen/PaperPages/HDRVideo/) | [Kalantari13](https://web.ece.ucsb.edu/~psen/PaperPages/HDRVideo/) |     |
| Deep HDR Video from Sequences with Alternating Exposures | [Eurographics-2019](https://people.engr.tamu.edu/nimak/Data/Eurographics19_HDRVideo_LoRes.pdf) |     |     | First DL HDR Video |
| Creating cinematic wide gamut HDR-video for the evaluation of tone mapping operators and HDR-displays | [Digital photography-2014](https://imago.org/wp-content/uploads/2014/10/images_pdfs_EDUCATION_Cinematic_HDR_Video.pdf) | [Porject](https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/) | [HDM-HDR-2014](https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/) | HDR video data |

### Tone Mapping

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| Zero-Shot Structure-Preserving Diffusion Model for High Dynamic Range Tone Mapping | [CVPR-2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Zero-Shot_Structure-Preserving_Diffusion_Model_for_High_Dynamic_Range_Tone_Mapping_CVPR_2024_paper.pdf) | [code](https://github.com/ZSDM-HDR/Zero-Shot-Diffusion-HDR) |     |     |
| Unsupervised HDR Image and Video Tone Mapping via Contrastive Learning | [TCSVT-2023](https://arxiv.org/pdf/2303.07327.pdf) | [UnCLTMO](https://github.com/cao-cong/UnCLTMO) | [UVTM](https://github.com/cao-cong/UnCLTMO) |     |

### Traditional HDRI

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss | [TPAMI 2015](https://joonyoung-cv.github.io/assets/paper/15_pami_robust_high.pdf) |     |     |     |
| HDR Deghosting: How to deal with saturation? | [CVPR 2013](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.7215&rep=rep1&type=pdf) |     |     |     |
| Robust patch-based HDR reconstruction of dynamic scenes | [TOG 2012](https://people.engr.tamu.edu/nimak/Data/SIGASIA12_HDR_PatchBasedReconstruction_LoRes.pdf) | [Code](https://web.ece.ucsb.edu/~psen/hdrvideo) |     |     |

### Other

| Title | Paper | Code | Dataset | Key Words |
| --- | --- | --- | --- | --- |
| Unsupervised Optical Flow Estimation for Differently Exposed Images in LDR Domain | [TCSVT-2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10058569&tag=1) | [LDRFlow](https://github.com/liuziyang123/LDRFlow) |     |     |
| Polarization Guided HDR Reconstruction via Pixel-Wise Depolarization | [TIP-2023](https://ieeexplore.ieee.org/document/10061479) |     |     |     |
| 1000 FPS HDR Video with a Spike-RGB Hybrid Camera | [CVPR-2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chang_1000_FPS_HDR_Video_With_a_Spike-RGB_Hybrid_Camera_CVPR_2023_paper.html) |     |     |     |
| Hybrid High Dynamic Range Imaging fusing Neuromorphic and Conventional Images | [TPAMI-2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10036136) | [NeurImg-HDR](https://github.com/hjynwa/NeurImg-HDR) |     |     |
| Text2Light: Zero-Shot Text-Driven HDR Panorama Generation | [TOG-2022](https://dl.acm.org/doi/abs/10.1145/3550454.3555447) | [Text2Light](https://github.com/FrozenBurning/Text2Light) |     |     |
| HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields | [ECCV-2022](https://arxiv.org/abs/2208.06787) | [HDR-Plenoxels](https://github.com/postech-ami/HDR-Plenoxels) |     |     |
| How To Cheat With Metrics in Single-Image HDR Reconstruction | [ICCV-2021](https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Eilertsen_How_To_Cheat_With_Metrics_in_Single-Image_HDR_Reconstruction_ICCVW_2021_paper.html) | /   |     |     |
| Neuromorphic Camera Guided High Dynamic Range Imaging | [CVPR-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_Neuromorphic_Camera_Guided_High_Dynamic_Range_Imaging_CVPR_2020_paper.pdf) |     |     |     |
| UnModNet: Learning to Unwrap a Modulo Image for High Dynamic Range Imaging | [NIPS-2020](https://proceedings.neurips.cc/paper/2020/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html) |     |     |     |

###

## HDR Challenges

| Title | Paper | Compeptition | Dataset |
| --- | --- | --- | --- |
| NTIRE 2022 Challenge on High Dynamic Range Imaging : Methods and Results | [CVPRW-2022](https://arxiv.org/pdf/2205.12633.pdf#:~:text=This%20manuscript%20focuses%20on%20the,and%20different%20sources%20of%20noise) | [Competition](https://data.vision.ee.ethz.ch/cvl/ntire22/) | NTIRE |
| NTIRE 2021 Challenge on High Dynamic Range Imaging: Dataset, Methods and Results | [CVPRW-2021](https://arxiv.org/abs/2106.01439) | [Competition](https://data.vision.ee.ethz.ch/cvl/ntire21/) | NTIRE |

## HDR Datasets

### HDR Image Datasets

| Dataset | Amount | Data type | GT  | Resolution | Details |
| --- | --- | --- | --- | --- | --- |
| [UltraFusion](https://openimaginglab.github.io/UltraFusion/) | 100 |     | Yes |     | real-captured under/over-exposed image pairs (up to 9 stops), DSLR+mobile |
| [Mobile-HDR](https://github.com/shuaizhengliu/Joint-HDRDN) | 115 dynamic+136 static | Real LDR + bracketing HDR | Yes | 4K  | shot by mobile phones, daytime+nighttime |
| [NTIRE-HDR](https://competitions.codalab.org/competitions/28162) | 1500 (training) + 60 (validation) + 201 (testing) | Real HDR + synthetic LDR | Yes | 1900 x 1060 | 29 scenes |
| [Kalantari *et al.*](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) | 74 (training) + 15 (testing) | Real LDR + bracketing HDR | Yes | 1500 x 1000 | Multi-exposure, dynamic scenes |
| [Sen *et al.*](https://web.ece.ucsb.edu/~psen/hdrvideo) | 8 (testing only) | Real | No  | 1350 x 900 | Multi-exposure, dynamic scenes, collected from HDR videos [1] |
| [Tursen *et al.*](https://user.ceng.metu.edu.tr/~akyuz/files/eg2016/index.html) | 16 (testing only) | Real | No  | 1024 x 682 | Multi-exposure, indoor and out door scenes |
| [Prabhakar *et al.*](https://val.cds.iisc.ac.in/HDR/ICCP19/) | 466 (training) + 116 (testing) | Real | Yes | 1-4 Megapixels | Dynamic scenes of 3-7 images (Download permission denied) |

### HDR Video Datasets

| Dataset | Published | Amount | Data type | GT  | Resolution | Details |
| --- | --- | --- | --- | --- | --- | --- |
| [Real-HDRV](https://github.com/yungsyu99/Real-HDRV) | CVPR 2024 | 500 scenes * (7-10) frames |     | Yes |     |     |
| [DeepHDRVideo](https://guanyingc.github.io/DeepHDRVideo/) | ICCV 2021 |     |     | Yes | 1500 x 1000 | Raw Res: 6000 x4000, |
| [Kalantari13](https://web.ece.ucsb.edu/~psen/PaperPages/HDRVideo/) | TOG 2013 |     |     | No  | 1280 x 720 |     |

**Reference**

[1][wang, l., & yoon, k. j. (2021). deep learning for hdr imaging: state-of-the-art and future trends. ieee transactions on pattern analysis and machine intelligence.](<https://arxiv.org/pdf/2110.10394.pdf>)

[2] Froehlich, J., Grandinetti, S., Eberhardt, B., Walter, S., Schilling, A., & Brendel, H. (2014, March). Creating cinematic wide gamut HDR-video for the evaluation of tone mapping operators and HDR-displays. In Digital photography X (Vol. 9023, pp. 279-288). SPIE.

## Interesting HDR Reading Materials

- [High Dynamic Range Imaging (second edition)](https://last.hit.bme.hu/download/firtha/video/HDR/Erik%20Reinhard,%20Wolfgang%20Heidrich,%20Paul%20Debevec,%20Sumanta%20Pattanaik,%20Greg%20Ward,%20Karol%20Myszkowski%20High%20Dynamic%20Range%20Imaging,%20Second%20Edition%20Acquisition,%20Display,%20and%20Image-Based%20Lighting%20%202010.pdf)

## See other useful HDR paperlists

- <https://github.com/vinthony/awesome-deep-hdr>
- <https://github.com/ytZhang99/Awesome-HDR>
- <https://github.com/lcybuzz/Low-Level-Vision-Paper-Record/blob/master/HDR.md>
