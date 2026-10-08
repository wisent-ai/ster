//! Grouping compared artifacts: average-linkage agglomeration on a
//! precomputed distance matrix, cut where the caller says, and the mean
//! silhouette that tells the caller how well that cut separates them.

/// Merge the two nearest groups — nearest by the mean distance between
/// their members — until `target` groups remain. Returns each item's group
/// in item order; groups are numbered by their earliest member.
pub(super) fn average(distance: &[Vec<f64>], target: usize) -> Vec<usize> {
    let mut groups: Vec<Vec<usize>> = distance.iter().enumerate().map(|(item, _)| vec![item]).collect();
    while groups.len() > target {
        let nearest = groups
            .iter()
            .enumerate()
            .flat_map(|(left, _)| {
                groups
                    .iter()
                    .enumerate()
                    .filter(move |(right, _)| *right > left)
                    .map(move |(right, _)| (left, right))
            })
            .min_by(|&(a, b), &(c, d)| {
                between(distance, &groups[a], &groups[b]).total_cmp(&between(distance, &groups[c], &groups[d]))
            })
            .expect("more groups than the target means at least two to merge");
        let (keep, absorb) = nearest;
        let absorbed = groups.remove(absorb);
        groups[keep].extend(absorbed);
    }
    let mut assignment = vec![groups.len(); distance.len()];
    for (group, members) in groups.iter().enumerate() {
        for &member in members {
            assignment[member] = group;
        }
    }
    assignment
}

/// Mean silhouette over the items that share their group with another and
/// have a nearer-or-farther group to be told apart from. `None` when no
/// item qualifies: every item in one group, or each alone in its own.
pub(super) fn silhouette(distance: &[Vec<f64>], assignment: &[usize]) -> Option<f64> {
    let scores: Vec<f64> = assignment
        .iter()
        .enumerate()
        .filter_map(|(item, &group)| {
            let own: Vec<usize> = members(assignment, group).into_iter().filter(|&other| other != item).collect();
            if own.is_empty() {
                return None;
            }
            let cohesion = mean_to(distance, item, &own);
            let separation = assignment
                .iter()
                .filter(|&&other| other != group)
                .copied()
                .collect::<std::collections::BTreeSet<usize>>()
                .into_iter()
                .map(|other| mean_to(distance, item, &members(assignment, other)))
                .min_by(f64::total_cmp)?;
            let scale = cohesion.max(separation);
            scale.is_normal().then(|| (separation - cohesion) / scale)
        })
        .collect();
    (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64)
}

fn members(assignment: &[usize], group: usize) -> Vec<usize> {
    assignment
        .iter()
        .enumerate()
        .filter(|&(_, &assigned)| assigned == group)
        .map(|(item, _)| item)
        .collect()
}

fn mean_to(distance: &[Vec<f64>], item: usize, others: &[usize]) -> f64 {
    others.iter().map(|&other| distance[item][other]).sum::<f64>() / others.len() as f64
}

fn between(distance: &[Vec<f64>], left: &[usize], right: &[usize]) -> f64 {
    let total: f64 = left
        .iter()
        .flat_map(|&a| right.iter().map(move |&b| distance[a][b]))
        .sum();
    total / (left.len() * right.len()) as f64
}
