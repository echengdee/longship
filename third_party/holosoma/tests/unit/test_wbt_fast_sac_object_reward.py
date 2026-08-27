from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_fast_sac_w_object


def test_object_fast_sac_strengthens_body_and_object_position_tracking() -> None:
    terms = g1_29dof_wbt_fast_sac_w_object.reward.terms

    assert terms["motion_relative_body_position_error_exp"].weight == 2.0
    assert terms["action_rate_l2"].weight == -0.1
    assert terms["motion_global_ref_position_error_exp"].weight == 0.5
    assert terms["object_global_ref_position_error_exp"].weight == 2.0
    assert terms["object_global_ref_orientation_error_exp"].weight == 1.0
    assert g1_29dof_wbt_fast_sac_w_object.algo.config.actor_learning_rate == 1.5e-4
