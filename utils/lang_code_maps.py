import os, sys

def vl107_to_fleurs_map():
    '''Map VL107 language codes to FLEURS language codes'''
    vl107_codes = os.listdir("/exp/jvillalba/corpora/voxlingua107")
    fl_codes = os.listdir("/export/common/data/corpora/fleurs/metadata")

    # vl107_to_fleurs_map = {}
    # for vl_code in vl107_codes:
    #     for fl_code in fl_codes:
    #         if vl_code == fl_code.split("_")[0]:
    #             vl107_to_fleurs_map[vl_code] = fl_code
    #             break
    vl107_to_fleurs_map = {'am': 'am_et', 'as': 'as_in', 'ar': 'ar_eg', 'be': 'be_by', 'az': 'az_az', 'bg': 'bg_bg', 'af': 'af_za', 'bn': 'bn_in', 'en': 'en_us', 'es': 'es_419', 'el': 'el_gr', 'et': 'et_ee', 'da': 'da_dk', 'cs': 'cs_cz', 'cy': 'cy_gb', 'de': 'de_de', 'ca': 'ca_es', 'ceb': 'ceb_ph', 'bs': 'bs_ba', 'gu': 'gu_in', 'fi': 'fi_fi', 'fr': 'fr_fr', 'gl': 'gl_es', 'fa': 'fa_ir', 'hu': 'hu_hu', 'hi': 'hi_in', 'hr': 'hr_hr', 'ha': 'ha_ng', 'hy': 'hy_am', 'km': 'km_kh', 'kk': 'kk_kz', 'ka': 'ka_ge', 'ja': 'ja_jp', 'id': 'id_id', 'kn': 'kn_in', 'it': 'it_it', 'is': 'is_is', 'ko': 'ko_kr', 'mi': 'mi_nz', 'my': 'my_mm', 'lv': 'lv_lv', 'mt': 'mt_mt', 'mk': 'mk_mk', 'lt': 'lt_lt', 'lb': 'lb_lu', 'ms': 'ms_my', 'mr': 'mr_in', 'lo': 'lo_la', 'mn': 'mn_mn', 'ln': 'ln_cd', 'ml': 'ml_in', 'ne': 'ne_np', 'pa': 'pa_in', 'oc': 'oc_fr', 'ps': 'ps_af', 'sd': 'sd_in', 'ro': 'ro_ro', 'pl': 'pl_pl', 'ru': 'ru_ru', 'nl': 'nl_nl', 'pt': 'pt_br', 'so': 'so_so', 'th': 'th_th', 'sw': 'sw_ke', 'sr': 'sr_rs', 'tg': 'tg_tj', 'ta': 'ta_in', 'sk': 'sk_sk', 'te': 'te_in', 'sn': 'sn_zw', 'sl': 'sl_si', 'sv': 'sv_se', 'ur': 'ur_pk', 'uk': 'uk_ua', 'yo': 'yo_ng', 'vi': 'vi_vn', 'tr': 'tr_tr', 'uz': 'uz_uz'}
    return vl107_to_fleurs_map
