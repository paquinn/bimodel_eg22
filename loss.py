def l1_objective(x, x0, num_real_params,
                mytrace, derivs, verts_locations, endpoints, lengths,
                selected_verts, not_selected,
                geosemantic_constraints, #type of constraint, 'COPLANAR[]', 'PARALLEL [[],[]]', 'COCENTRIC []'
                weights,
                per_vertex_weight, per_parameter_weight,
                thresh = 0.001):

    # Parameter loss
    # normalize both vectors!
    #normalized_x0 = x0[0:num_real_params] #.normalize()
    d_p = (x[0:num_real_params]-x0[0:num_real_params])*per_parameter_weight
    l1_p = np.linalg.norm(d_p, 1) / np.linalg.norm(per_parameter_weight, 1)

    # Vertex deformation
    d_v = (mytrace(x)[not_selected] - verts_locations[not_selected]).flatten()*(per_vertex_weight[not_selected].flatten())
    nv = np.linalg.norm(per_vertex_weight, 1)
    l1_v_x, l1_v_y, l1_v_z = np.linalg.norm(d_v[0::3], 1) / nv, np.linalg.norm(d_v[1::3], 1) / nv,  np.linalg.norm(d_v[2::3], 1) / nv
    # l1_v_x, l1_v_y, l1_v_z = np.linalg.norm(d_v[0::3], 1) / len(not_selected), np.linalg.norm(d_v[1::3], 1) / len(not_selected),  np.linalg.norm(d_v[2::3], 1) / len(not_selected)
    #ave_move_x, ave_move_y, ave_move_z = d_v[0::3]-l1_v_x, d_v[1::3]-l1_v_y, d_v[2::3]-l1_v_z
    #ave_move_x, ave_move_y, ave_move_z = norm(ave_move_x, 1), norm(ave_move_y, 1), norm(ave_move_z, 1)

    # Edge deformation
    uvs = mytrace(x)[endpoints[:,0]] - mytrace(x)[endpoints[:,1]]
    new_squared_edge_lengths = np.sum(uvs**2, axis=1)
    diff = new_squared_edge_lengths - lengths**2
    #diff = diff * per_edge_weights
    l1_e = np.linalg.norm(diff, 1) / len(endpoints)

    # Selected vertices
    constraint = (mytrace(x)[selected_verts] - verts_locations[selected_verts]).flatten()
    l2_norm = np.linalg.norm(constraint, 2)**2

    # Geosemantic constraints
    # TODO
    geosemantic_loss = 0.0
    #for const in geosemantic_constraints:
    #    if const[0] == 'COPLANAR':
    #        elements = np.array(const[1].to_list())
    #        if len(elements) > 3:
    #            geosemantic_loss += np.linalg.norm( np.sum( np.cross( trace(x)[elements[1]]-trace(x)[elements[0]], trace(x)[elements[2]]-trace(x)[elements[0]]) * (trace(x)[elements[3:]]-trace(x)[elements[0]]), axis=1) )

    loss = weights.dot([l2_norm, l1_p, (l1_v_x+l1_v_y+l1_v_z), l1_e, geosemantic_loss])
    print(loss)
    return loss

def der_l1_objective(x, x0, num_real_params,
                    mytrace, derivs, verts_locations, endpoints, lengths,
                    selected_verts, not_selected,
                    geosemantic_constraints, #type of constraint, 'COPLANAR[]', 'PARALLEL [[],[]]', 'COCENTRIC []'
                    weights,
                    per_vertex_weight, per_parameter_weight,
                    thresh = 0.001):
    # dp
    s_p = np.sign(x[0:num_real_params]-x0[0:num_real_params])
    d_p = np.concatenate((s_p*per_parameter_weight, np.zeros(len(x0)-num_real_params))) / np.linalg.norm(per_parameter_weight, 1) # / len(x0[0:num_real_params])

    #dv
    s_v = np.sign((mytrace(x)[not_selected] - verts_locations[not_selected]).flatten())*(per_vertex_weight[not_selected].flatten())
    s_v = s_v.reshape((3*len(not_selected), 1))
    d_v = np.array( derivs(x)[not_selected].reshape(3*len(not_selected), len(x0)))
    nv = np.linalg.norm(per_vertex_weight, 1)
    d_v = np.sum(s_v*d_v, axis=0) / nv
    #d_v = np.sum(s_v*d_v, axis=0) / len(not_selected)

    # de
    uvs = mytrace(x)[endpoints[:,0]] - mytrace(x)[endpoints[:,1]]
    new_squared_edge_lengths = np.sum(uvs**2, axis=1)
    diff = new_squared_edge_lengths - lengths**2
    s_e = np.sign(diff) #* per_edge_weights
    uvs = uvs.reshape([uvs.shape[0], uvs.shape[1], 1])
    d_in = np.sum( uvs * (derivs(x)[endpoints[:,0]]-derivs(x)[endpoints[:,1]]), axis=1)
    d_e = np.sum(s_e[:, np.newaxis] * d_in, axis=0) / len(endpoints)

    d_selected = np.array( derivs(x)[selected_verts].reshape(3*len(selected_verts), len(x)))
    d_s_in = (mytrace(x)[selected_verts] - verts_locations[selected_verts]).flatten()
    d_selected = 2.0 * d_s_in[:, np.newaxis] * d_selected

    d_g = 0

    return weights[0]*np.sum(d_selected, axis=0) + weights[1]*d_p + weights[2]*d_v + weights[3]*d_e
