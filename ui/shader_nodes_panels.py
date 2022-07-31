import bpy


# Only registered in Blender 2.80+
class ShaderNodesOverridePanel(bpy.types.Panel):
    bl_label = "Material Combiner Override"
    bl_idname = 'SMC_PT_Shader_Nodes_Override'
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'MatCombiner'

    @classmethod
    def poll(cls, context):
        # Don't need to check for context.space_data.type == 'NODE_EDITOR' because it's the Panel's bl_space_type
        space_data = context.space_data
        # Don't show when editing a group node.
        if space_data.node_tree != space_data.edit_tree:
            return False
        # Only show the panel in the shader nodes window and only if the current material has nodes.
        if space_data.tree_type == 'ShaderNodeTree':
            if hasattr(context, 'material'):
                mat = context.material
                if mat:
                    node_tree = mat.node_tree
                    if node_tree:
                        return node_tree.nodes
        return False

    def draw(self, context):
        layout = self.layout
        layout.label(text="Override search start node")
        row = layout.row()
        # draw button to set active node to material combiner override
        row.operator('smc.shader_nodes_set_active_as_override')
        # draw button to select material combiner override (the operator should disable the button via its poll
        #   function when there is no current override)
        row.operator('smc.shader_nodes_set_override_as_active')

        row = layout.row()
        # draw button to clear the material combiner override
        row.operator('smc.shader_nodes_clear_override')
        # draw button to frame (view) the material combiner override
        row.operator('smc.shader_nodes_frame_override')

        layout.separator()
        mat = context.material
        override_name = mat.smc_override_node_name
        if override_name:
            layout.label(text="Override node:")
            node_tree = mat.node_tree
            override_node = node_tree.nodes.get(mat.smc_override_node_name)
            box = layout.box()
            if override_node:
                # Display the same displayed label of the node as an expandable header
                wm = context.window_manager
                box.prop(wm, "smc_override_node_toggle_full_view", emboss=False,
                         icon="TRIA_DOWN" if wm.smc_override_node_toggle_full_view else "TRIA_RIGHT",
                         text=override_node.label if override_node.label else override_node.bl_label)
                # If expanded show the full view of the node
                if wm.smc_override_node_toggle_full_view:
                    # Draw the node in the same way as the Surface panel in Material Properties, but without the input
                    # the override_node's output is connected to
                    box.template_node_view(mat.node_tree, override_node, None)
            else:
                box.label(text="Node '{}' not found".format(override_name))
        else:
            layout.label(text="No override set")
            layout.label(text="Using active output node(s)")
