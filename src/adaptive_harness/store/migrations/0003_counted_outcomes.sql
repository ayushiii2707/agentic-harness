-- Evaluation findings F4 and F5.
-- F5: a trial records the profile that served it, so later feedback is credited to that profile, not the newest one.
ALTER TABLE trial ADD COLUMN profile_id INTEGER REFERENCES profile(id);
-- F4: at most one evaluation per trial counts as evidence, enforced by the database (atomic across threads and
-- processes). Later evaluations of the same trial are kept as records with counted = 0.
ALTER TABLE evaluation ADD COLUMN counted INTEGER NOT NULL DEFAULT 0;
UPDATE evaluation SET counted = 1
 WHERE status IN ('success','failure','partial') AND score IS NOT NULL
   AND id = (SELECT MIN(e2.id) FROM evaluation e2
              WHERE e2.trial_id = evaluation.trial_id AND e2.status IN ('success','failure','partial')
                AND e2.score IS NOT NULL);
CREATE UNIQUE INDEX ux_evaluation_counted ON evaluation(trial_id) WHERE counted = 1;
