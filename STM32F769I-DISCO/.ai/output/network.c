/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2024-10-28T16:31:11+0000
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0xe66b1f6b33d465f86ffd74c494c14ba7"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2024-10-28T16:31:11+0000"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 784, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 10, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100352, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 640, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 10, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 1424, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 448, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 114, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015657147392630577f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025485532358288765f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04691920056939125f),
    AI_PACK_INTQ_ZP(32)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000650793663226068f, 0.005871832836419344f, 0.004435253329575062f, 0.004744454752653837f, 0.0006491172825917602f, 0.003090388374403119f, 0.0038523052353411913f, 0.007107751909643412f, 0.000693467038217932f, 0.004294286482036114f, 0.004411862697452307f, 0.006852064281702042f, 0.0041768793016672134f, 0.0037726950831711292f, 0.0007119184592738748f, 0.004142250400036573f, 0.00444644084200263f, 0.00497982744127512f, 0.0048788744024932384f, 0.0051838126964867115f, 0.008211437612771988f, 0.005817683879286051f, 0.00600328017026186f, 0.0006626714020967484f, 0.005550360307097435f, 0.005165937356650829f, 0.005009850021451712f, 0.004422267898917198f, 0.005545718129724264f, 0.0006589894182980061f, 0.005634465254843235f, 0.006474880967289209f, 0.007299339864403009f, 0.004918360151350498f, 0.007268483750522137f, 0.0006856120890006423f, 0.0043423674069345f, 0.0006655650213360786f, 0.0006689192960038781f, 0.0006820308044552803f, 0.000651652691885829f, 0.004822288174182177f, 0.004974777344614267f, 0.008193494752049446f, 0.006059772800654173f, 0.0006564180948771536f, 0.0038532286416739225f, 0.005053901579231024f, 0.006216145120561123f, 0.004820314235985279f, 0.003860146040096879f, 0.0006515983259305358f, 0.00688743544742465f, 0.0006941146566532552f, 0.004914997145533562f, 0.0007200120599009097f, 0.007551829796284437f, 0.0006508848164230585f, 0.004186863545328379f, 0.004112017806619406f, 0.005094155669212341f, 0.0006439252174459398f, 0.0006511532701551914f, 0.0040484401397407055f, 0.003359321504831314f, 0.0007143449620343745f, 0.008869167417287827f, 0.005160574335604906f, 0.008022031746804714f, 0.005160832777619362f, 0.0006502725300379097f, 0.005420939531177282f, 0.0006991254049353302f, 0.00071424909401685f, 0.0050554643385112286f, 0.00636120093986392f, 0.004771013278514147f, 0.0029904376715421677f, 0.000688424042891711f, 0.0037899683229625225f, 0.0006900503067299724f, 0.004926631692796946f, 0.004459646064788103f, 0.005522849038243294f, 0.006209916900843382f, 0.007221275474876165f, 0.0006574215949513018f, 0.0007055060123093426f, 0.0006497318390756845f, 0.0006819296977482736f, 0.0006547015509568155f, 0.0042406488209962845f, 0.005697629880160093f, 0.003060139250010252f, 0.00725291483104229f, 0.004497351124882698f, 0.006798930466175079f, 0.00682430574670434f, 0.0006913075922057033f, 0.005728005897253752f, 0.004613245837390423f, 0.0006797973765060306f, 0.003556421957910061f, 0.0032617084216326475f, 0.0060181948356330395f, 0.004146253690123558f, 0.0006513134576380253f, 0.004124733153730631f, 0.004532450344413519f, 0.0006777874659746885f, 0.006641875486820936f, 0.0032794689759612083f, 0.00603551184758544f, 0.0006594997830688953f, 0.005628996528685093f, 0.004429022781550884f, 0.0006514647975564003f, 0.0006517787696793675f, 0.0056124343536794186f, 0.004762687720358372f, 0.00384600181132555f, 0.0006710138986818492f, 0.000677969423122704f, 0.0030978003051131964f, 0.004520575515925884f, 0.008831807412207127f, 0.004036935046315193f, 0.003986888099461794f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0046560815535485744f, 0.005360245238989592f, 0.004314982332289219f, 0.0066299461759626865f, 0.002019635634496808f, 0.00501110078766942f, 0.005218518432229757f, 0.004097759258002043f, 0.001725216512568295f, 0.0018308673752471805f, 0.0039449124597013f, 0.005953003652393818f, 0.004183558281511068f, 0.0040244972333312035f, 0.004759497474879026f, 0.0018417108803987503f, 0.004395986907184124f, 0.004272449761629105f, 0.004328022710978985f, 0.0017549896147102118f, 0.0019077215110883117f, 0.003872370347380638f, 0.004283739719539881f, 0.0033664603251963854f, 0.005507763009518385f, 0.005177340004593134f, 0.0037884803023189306f, 0.006510674487799406f, 0.004931424278765917f, 0.005454453639686108f, 0.004094907082617283f, 0.005390876438468695f, 0.0036532084923237562f, 0.002718632575124502f, 0.0019121775403618813f, 0.004456347320228815f, 0.003934663720428944f, 0.0052132923156023026f, 0.004816322587430477f, 0.004600069485604763f, 0.004324212204664946f, 0.0045369160361588f, 0.007198837120085955f, 0.005042914301156998f, 0.0016061272472143173f, 0.004113981034606695f, 0.0042090704664587975f, 0.004752648063004017f, 0.006040455307811499f, 0.005660370923578739f, 0.005823832470923662f, 0.005411532241851091f, 0.004284009337425232f, 0.004637561738491058f, 0.003384525654837489f, 0.0045598773285746574f, 0.005733346100896597f, 0.00417803879827261f, 0.005428288597613573f, 0.004389481153339148f, 0.003731498960405588f, 0.004421670455485582f, 0.004451419692486525f, 0.006002714391797781f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008195655420422554f, 0.007444640621542931f, 0.0062379552982747555f, 0.00581892067566514f, 0.0070450385101139545f, 0.006153288763016462f, 0.006804723292589188f, 0.006320445332676172f, 0.0053804293274879456f, 0.005254882387816906f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921391908079386f),
    AI_PACK_INTQ_ZP(-128)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_scratch0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 448, 1, 1), AI_STRIDE_INIT(4, 2, 2, 896, 896),
  1, &_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_Relu_1_output_0_output_array, &_Relu_1_output_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &output_QuantizeLinear_Input_output_array, &output_QuantizeLinear_Input_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 784, 128, 1, 1), AI_STRIDE_INIT(4, 1, 784, 100352, 100352),
  1, &_Relu_output_0_weights_array, &_Relu_output_0_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 114, 1, 1), AI_STRIDE_INIT(4, 2, 2, 228, 228),
  1, &output_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_Relu_output_0_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_weights, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 1, 128, 8192, 8192),
  1, &_Relu_1_output_0_weights_array, &_Relu_1_output_0_weights_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Relu_1_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 64, 10, 1, 1), AI_STRIDE_INIT(4, 1, 64, 640, 640),
  1, &output_QuantizeLinear_Input_weights_array, &output_QuantizeLinear_Input_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &output_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_scratch0, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1424, 1, 1), AI_STRIDE_INIT(4, 2, 2, 2848, 2848),
  1, &_Relu_output_0_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 28, 28, 1), AI_STRIDE_INIT(4, 1, 1, 28, 784),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  input_output0, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 784, 1, 1), AI_STRIDE_INIT(4, 1, 1, 784, 784),
  1, &input_output_array, &input_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &output_QuantizeLinear_Input_weights, &output_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  output_QuantizeLinear_Input_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &output_QuantizeLinear_Input_chain,
  NULL, &output_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_1_output_0_weights, &_Relu_1_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_1_output_0_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_1_output_0_chain,
  NULL, &output_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_output_0_weights, &_Relu_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 10,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_Relu_output_0_chain,
  NULL, &_Relu_1_output_0_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 109992, 1, 1),
    109992, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3760, 1, 1),
    3760, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &output_QuantizeLinear_Input_output),
  &_Relu_output_0_layer, 0xe51c93ed, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 109992, 1, 1),
      109992, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3760, 1, 1),
      3760, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &output_QuantizeLinear_Input_output),
  &_Relu_output_0_layer, 0xe51c93ed, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    _Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Relu_1_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Relu_1_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    output_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    output_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    input_output_array.data = AI_PTR(g_network_activations_map[0] + 2848);
    input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2848);
    _Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3632);
    _Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3632);
    _Relu_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2848);
    _Relu_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2848);
    output_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 2912);
    output_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2912);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 100352);
    _Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 100352);
    _Relu_1_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_1_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 100864);
    _Relu_1_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 100864);
    _Relu_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 109056);
    _Relu_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 109056);
    output_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    output_QuantizeLinear_Input_weights_array.data = AI_PTR(g_network_weights_map[0] + 109312);
    output_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 109312);
    output_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    output_QuantizeLinear_Input_bias_array.data = AI_PTR(g_network_weights_map[0] + 109952);
    output_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 109952);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 109386,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xe51c93ed,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 109386,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xe51c93ed,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

