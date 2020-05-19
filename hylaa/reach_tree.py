
class ReachTreeTransition(object):
        def __init__(self, succ_node):
            self.succ_node = succ_node


class DiscreteTransition(ReachTreeTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, ReachTreeNode)
        ReachTreeTransition.__init__(self,  succ_node)


class ContinuousTransition(ReachTreeTransition):

    def __init__(self, succ_node):
        assert isinstance(succ_node, ReachTreeNode)
        ReachTreeTransition.__init__(self, succ_node)


class ReachTreeNode(object):

    def __init__(self, state, unsafe):

        self.state = state
        self.cont_transition = None   # There is at most one continuos successor
        self.disc_transitions = []
        self.error = unsafe

    def new_transition(self, succ_node):

        if self.state.mode.name == succ_node.state.mode.name:
            c_transition = ContinuousTransition(succ_node)
            self.cont_transition = c_transition
        else:
            d_transition = DiscreteTransition(succ_node)
            self.disc_transitions.append(d_transition)


class ReachTree(object):

    def __init__(self):

        # List of all the nodes in the tree
        self.nodes = []

        # To keep track of the nodes yet to be visited while creating the tree
        self.cont_leaf_nodes = []
        self.disc_leaf_nodes = []

    def get_node(self, state, cont_or_disc):

        if cont_or_disc == 0:  # Continuous
            if len(self.cont_leaf_nodes) > 0:
                for index in range(len(self.cont_leaf_nodes)):
                    node = self.cont_leaf_nodes[index]
                    if (node.state.mode.name == state.mode.name) and (node.state.total_steps == state.total_steps):
                        del self.cont_leaf_nodes[index]
                        return node

        elif cont_or_disc == 1:
            if len(self.cont_leaf_nodes) > 0:
                del self.cont_leaf_nodes[:]

            # if it is the first node
            if len(self.disc_leaf_nodes) == 0:
                node = ReachTreeNode(state.clone(), False)
                self.nodes.append(node)
                return node
            else:
                for index in range(len(self.disc_leaf_nodes)):
                    node = self.disc_leaf_nodes[index]
                    if (node.state.mode.name == state.mode.name) and (node.state.total_steps == state.total_steps):
                        del self.disc_leaf_nodes[index]
                        return node

    def add_node(self, state, cont_or_disc, unsafe=False):

        node = ReachTreeNode(state, unsafe)
        if cont_or_disc == 0:  # 0 means continuous successor
            self.cont_leaf_nodes.append(node)
            # print "Continuous node: '{}' in location '{}'".format(state.total_steps, state.mode.name)
        elif cont_or_disc == 1:
            self.disc_leaf_nodes.append(node)
            # print "discrete node: '{}' in location '{}'".format(state.total_steps, state.mode.name)
        self.nodes.append(node)
        return node
